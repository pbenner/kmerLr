/* Copyright (C) 2019 Philipp Benner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type featureSelector struct {
  ClassWeights [2]float64
  Labels        []bool
  Kmers           KmerClassList
  KmersMap        map[KmerClassId]int
  Index         []int
  Names         []string
  IndexMap        map[int]int
  Transform       TransformFull
  Cooccurrence    bool
  N               int
  M               int
  Epsilon         float64
  Pool            threadpool.ThreadPool
}

/* -------------------------------------------------------------------------- */

func newFeatureSelector(config Config, kmers KmerClassList, index []int, names []string, cooccurrence bool, labels []bool, transform TransformFull, class_weights [2]float64, m, n int, epsilon float64) featureSelector {
  kmersMap := make(map[KmerClassId]int)
  for i := 0; i < len(kmers); i++ {
    kmersMap[kmers[i].KmerClassId] = i
  }
  indexMap := make(map[int]int)
  for i := 0; i < len(index); i++ {
    indexMap[index[i]] = i
  }
  r := featureSelector{
    Kmers       : kmers,
    KmersMap    : kmersMap,
    Index       : index,
    Names       : names,
    IndexMap    : indexMap,
    Transform   : transform,
    Labels      : labels,
    ClassWeights: class_weights,
    Cooccurrence: cooccurrence,
    // number of features to select
    N           : n,
    // data dimension (without co-occurrences)
    M           : m,
    Epsilon     : epsilon,
    Pool        : config.PoolLR }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) Select(data []ConstVector, theta []float64, features FeatureIndices, kmers KmerClassList, index []int, names []string, lambda float64, debug bool) (*featureSelection, float64, bool) {
  if obj.M != data[0].Dim()-1 {
    panic("internal error")
  }
  gd := []float64{}
  ok := false
  // copy all features i with theta_{i+1} != 0
  t, c, b := obj.restoreNonzero(theta, features, kmers, index)
  // compute gradient for selecting new features
  g_ := obj.gradient(data, t)[1:]
  if debug {
    gd = make([]float64, len(g_))
    copy(gd, g_)
  }
  // sort gradient entries with respect to absolute values
  g, i := NLargestAbsFloat64(g_, 2*obj.N)
  // add new features
  for k := 0; k < len(i); k++ {
    if c >= obj.N {
      break
    }
    if !b[i[k]+1] && g[k] != 0.0 {
      // feature was previously zero
      ok        = true
      b[i[k]+1] = true
      c        += 1
    }
  }
  // add old features if not enough were found
  for k := 0; k < len(i); k++ {
    if c >= obj.N {
      break
    }
    if !b[i[k]+1] {
      // feature was previously used
      b[i[k]+1] = true
      c        += 1
    }
  }
  if c > obj.N {
    ok = true
  }
  if debug {
    for i := 1; i < len(b); i++ {
      fmt.Printf("gradient: %v %v", i-1, gd[i-1])
      if b[i] {
        fmt.Println(" *")
      } else {
        fmt.Println()
      }
    }
  }
  l          := obj.computeLambda(b, g, g_)
  k, x, s, f := obj.selectKmers(b)
  tr         := obj.Transform.Select(b)
  return &featureSelection{obj, k, x, s, f, t, tr, b, c}, l, ok || (obj.Epsilon > 0.0 && math.Abs(lambda - l) >= obj.Epsilon)
}

func (obj featureSelector) SelectFixed(data []ConstVector, theta []float64, features FeatureIndices, kmers KmerClassList, index []int, names []string, l float64, max_features int) (*featureSelection, bool) {
  if obj.M != data[0].Dim()-1 {
    panic("internal error")
  }
  ok := false
  // copy all features i with theta_{i+1} != 0
  t, c, b := obj.restoreNonzero(theta, features, kmers, index)
  // compute gradient for selecting new features
  g := obj.gradient(data, t)
  // add new features
  for k := 1; k < len(g); k++ {
    if math.Abs(g[k]) >= l && b[k] == false {
      ok   = true
      b[k] = true
      c   += 1
    }
    if max_features > 0 && c >= max_features && ok {
      break
    }
  }
  k, x, s, f := obj.selectKmers(b)
  tr         := obj.Transform.Select(b)
  return &featureSelection{obj, k, x, s, f, t, tr, b, c}, ok
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) alloc(theta []float64) ([]bool, []float64) {
  t := []float64(nil)
  b := []bool   (nil)
  m := obj.M
  if obj.Cooccurrence {
    t = make([]float64, CoeffIndex(m).Dim())
    b = make([]bool   , CoeffIndex(m).Dim())
  } else {
    t = make([]float64, m+1)
    b = make([]bool   , m+1)
  }
  b[0] = true
  t[0] = theta[0]
  return b, t
}

func (obj featureSelector) computeLambda(b []bool, g, g_ []float64) float64 {
  if obj.N > len(g) {
    return 0.0
  }
  v := math.Abs(g[obj.N-1])
  w := 0.0
  // loop over unsorted gradient
  for k := 0; k < len(g_); k++ {
    if t := math.Abs(g_[k]); t > w && t < v {
      w = t
    }
  }
  return (v+w)/2.0
}

func (obj featureSelector) restoreNonzero(theta []float64, features FeatureIndices, kmers KmerClassList, index []int) ([]float64, int, []bool) {
  b, t := obj.alloc(theta)
  c    := 0
  m    := obj.M
  if len(obj.Index) > 0 {
    for i, feature := range features {
      j := CoeffIndex(m).Ind2Sub(obj.IndexMap[index[feature[0]]], obj.IndexMap[index[feature[1]]])
      if theta[i+1] != 0.0 {
        t[j] = theta[i+1]
        b[j] = true
        c   += 1
      }
    }
  }
  if len(obj.Kmers) > 0 {
    for i, feature := range features {
      j := CoeffIndex(m).Ind2Sub(obj.KmersMap[kmers[feature[0]].KmerClassId], obj.KmersMap[kmers[feature[1]].KmerClassId])
      if theta[i+1] != 0.0 {
        t[j] = theta[i+1]
        b[j] = true
        c   += 1
      }
    }
  }
  return t, c, b
}

func (obj featureSelector) gradient(data []ConstVector, theta []float64) []float64 {
  lr := logisticRegression{}
  lr.Theta        = theta
  lr.ClassWeights = obj.ClassWeights
  lr.Cooccurrence = obj.Cooccurrence
  lr.Pool         = obj.Pool
  lr.Transform    = obj.Transform
  return lr.Gradient(nil, data, obj.Labels)
}

func (obj featureSelector) selectKmers(b []bool) (KmerClassList, []int, []string, FeatureIndices) {
  r := KmerClassList{}
  x := []int{}
  s := []string{}
  f := FeatureIndices{}
  m := obj.M
  z := make([]bool, m)
  i := make([]int , m)
  for j := 1; j < len(b); j++ {
    if b[j] {
      i1, i2 := CoeffIndex(m).Sub2Ind(j-1)
      z[i1] = true
      z[i2] = true
    }
  }
  if len(obj.Kmers) != 0 && len(obj.Index) != 0 {
    panic("internal error")
  }
  if len(obj.Kmers) != 0 {
    for k := 0; k < m; k++ {
      if z[k] {
        i[k] = len(r)
        r    = append(r, obj.Kmers[k])
      }
    }
    for j := 1; j < len(b); j++ {
      if b[j] {
        i1, i2 := CoeffIndex(m).Sub2Ind(j-1)
        f = append(f, [2]int{i[i1], i[i2]})
      }
    }
  }
  if len(obj.Index) != 0 {
    for k := 0; k < m; k++ {
      if z[k] {
        i[k] = len(x)
        x    = append(x, obj.Index[k])
        if len(obj.Names) > 0 {
          s  = append(s, obj.Names[k])
        }
      }
    }
    for j := 1; j < len(b); j++ {
      if b[j] {
        i1, i2 := CoeffIndex(m).Sub2Ind(j-1)
        f = append(f, [2]int{i[i1], i[i2]})
      }
    }
  }
  return r, x, s, f
}

/* -------------------------------------------------------------------------- */

type featureSelection struct {
  featureSelector
  kmers     KmerClassList
  index   []int
  names   []string
  features  FeatureIndices
  theta   []float64
  transform Transform
  b       []bool
  c         int
}

/* -------------------------------------------------------------------------- */

func (obj *featureSelection) Theta() DenseFloat64Vector {
  r := []float64{}
  for i := 0; i < len(obj.b); i++ {
    if obj.b[i] {
      r = append(r, obj.theta[i])
    }
  }
  return NewDenseFloat64Vector(r)
}

func (obj *featureSelection) Data(config Config, data_dst, data []ConstVector) {
  k := []int{}
  m := obj.featureSelector.M
  // remap data indices
  for j := 0; j < len(obj.b); j++ {
    if obj.b[j] {
      k = append(k, j)
    }
  }
  if len(k) != obj.c+1 {
    panic("internal error")
  }
  for i_ := 0; i_ < len(data); i_++ {
    i := []int    {}
    v := []float64{}
    for j1, j2 := range k {
      if j2 >= data[i_].Dim() {
        i1, i2 := CoeffIndex(m).Sub2Ind(j2-1)
        if value := data[i_].Float64At(i1+1)*data[i_].Float64At(i2+1); value != 0.0 {
          i = append(i, j1)
          v = append(v, value)
        }
      } else {
        if value := data[i_].Float64At(j2); value != 0.0 {
          i = append(i, j1)
          v = append(v, value)
        }
      }
    }
    // resize slice and restrict capacity
    i = append([]int    {}, i[0:len(i)]...)
    v = append([]float64{}, v[0:len(v)]...)
    data_dst[i_] = UnsafeSparseConstFloat64Vector(i, v, obj.c+1)
  }
}

func (obj *featureSelection) Kmers() KmerClassList {
  return obj.kmers
}

func (obj *featureSelection) Index() []int {
  return obj.index
}

func (obj *featureSelection) Names() []string {
  return obj.names
}

func (obj *featureSelection) Features() FeatureIndices {
  return obj.features
}

func (obj *featureSelection) Transform() Transform {
  return obj.transform
}
