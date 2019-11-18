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

//import   "fmt"
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
  Transform       TransformFull
  Cooccurrence    bool
  N               int
  Epsilon         float64
  Pool            threadpool.ThreadPool
}

/* -------------------------------------------------------------------------- */

func newFeatureSelector(config Config, kmers KmerClassList, cooccurrence bool, labels []bool, transform TransformFull, class_weights [2]float64, n int, epsilon float64) featureSelector {
  m := make(map[KmerClassId]int)
  for i := 0; i < len(kmers); i++ {
    m[kmers[i].KmerClassId] = i
  }
  r := featureSelector{
    Kmers       : kmers,
    KmersMap    : m,
    Transform   : transform,
    Labels      : labels,
    ClassWeights: class_weights,
    Cooccurrence: cooccurrence,
    N           : n,
    Epsilon     : epsilon,
    Pool        : config.Pool }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) Select(data []ConstVector, theta []float64, features FeatureIndices, kmers KmerClassList, lambda float64) (featureSelection, float64, bool) {
  ok := false
  // copy all features i with theta_{i+1} != 0
  t, c, b := obj.restoreNonzero(theta, features, kmers)
  // compute gradient for selecting new features
  g_ := obj.gradient(data, t)
  // sort gradient entries with respect to absolute values
  g, i := NLargestAbsFloat64(g_, 2*obj.N)
  g_    = nil
  // add new features
  for k := 1; k < len(i); k++ {
    if c >= obj.N {
      break
    }
    if !b[i[k]] && g[k] != 0.0 {
      // feature was previously zero
      ok      = true
      b[i[k]] = true
      c      += 1
    }
  }
  l    := obj.computeLambda(b, g, i)
  k, f := obj.selectKmers(b)
  tr   := obj.Transform.Select(b)
  return featureSelection{obj, k, f, t, tr, b, c}, l, ok || (obj.Epsilon > 0.0 && math.Abs(lambda - l) >= obj.Epsilon)
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) computeLambda(b []bool, g []float64, i []int) float64 {
  if len(g) > obj.N+1 {
    return (math.Abs(g[obj.N]) + math.Abs(g[obj.N+1]))/2.0
  } else {
    return 0.0
  }
}

func (obj featureSelector) restoreNonzero(theta []float64, features FeatureIndices, kmers KmerClassList) ([]float64, int, []bool) {
  t := []float64(nil)
  b := []bool   (nil)
  c := 0
  m := len(obj.Kmers)
  if obj.Cooccurrence {
    t = make([]float64, CoeffIndex(m).Dim())
    b = make([]bool   , CoeffIndex(m).Dim())
  } else {
    t = make([]float64, m+1)
    b = make([]bool   , m+1)
  }
  b[0] = true
  t[0] = theta[0]
  for i, feature := range features {
    j := CoeffIndex(m).Ind2Sub(obj.KmersMap[kmers[feature[0]].KmerClassId], obj.KmersMap[kmers[feature[1]].KmerClassId])
    if theta[i+1] != 0.0 {
      t[j] = theta[i+1]
      b[j] = true
      c   += 1
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
  return lr.Gradient(nil, data, obj.Labels, nil)
}

func (obj featureSelector) selectKmers(b []bool) (KmerClassList, FeatureIndices) {
  r := KmerClassList{}
  f := FeatureIndices{}
  m := len(obj.Kmers)
  z := make([]bool, len(obj.Kmers))
  i := make([]int , len(obj.Kmers))
  for j := 1; j < len(b); j++ {
    if b[j] {
      i1, i2 := CoeffIndex(m).Sub2Ind(j-1)
      z[i1] = true
      z[i2] = true
    }
  }
  for k := 0; k < len(obj.Kmers); k++ {
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
  return r, f
}

/* -------------------------------------------------------------------------- */

type featureSelection struct {
  featureSelector
  kmers     KmerClassList
  features  FeatureIndices
  theta   []float64
  transform Transform
  b       []bool
  c         int
}

/* -------------------------------------------------------------------------- */

func (obj featureSelection) Theta() DenseBareRealVector {
  r := []float64{}
  for i := 0; i < len(obj.b); i++ {
    if obj.b[i] {
      r = append(r, obj.theta[i])
    }
  }
  return NewDenseBareRealVector(r)
}

func (obj featureSelection) Data(config Config, data_dst, data []ConstVector) {
  k := []int{}
  // remap data indices
  for j := 0; j < len(obj.b); j++ {
    if obj.b[j] {
      k = append(k, j)
    }
  }
  for i_ := 0; i_ < len(data); i_++ {
    i := []int    {}
    v := []float64{}
    for j1, j2 := range k {
      if value := data[i_].ValueAt(j2); value != 0.0 {
        i = append(i, j1)
        v = append(v, value)
      }
    }
    // resize slice and restrict capacity
    i = append([]int    {}, i[0:len(i)]...)
    v = append([]float64{}, v[0:len(v)]...)
    data_dst[i_] = UnsafeSparseConstRealVector(i, v, obj.c+1)
  }
  obj.transform.Apply(config, data_dst)
}

func (obj featureSelection) Kmers() KmerClassList {
  return obj.kmers
}

func (obj featureSelection) Features() FeatureIndices {
  return obj.features
}

func (obj featureSelection) Transform() Transform {
  return obj.transform
}
