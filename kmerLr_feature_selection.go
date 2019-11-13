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

/* -------------------------------------------------------------------------- */

type featureSelector struct {
  ClassWeights [2]float64
  Labels        []bool
  Kmers           KmerClassList
  KmersMap        map[KmerClassId]int
  Transform       Transform
  Cooccurrence    bool
  N               int
  Epsilon         float64
}

/* -------------------------------------------------------------------------- */

func newFeatureSelector(kmers KmerClassList, cooccurrence bool, labels []bool, transform Transform, class_weights [2]float64, n int, epsilon float64) featureSelector {
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
    Epsilon     : epsilon }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) Select(data []ConstVector, theta []float64, features FeatureIndices, kmers KmerClassList, lambda float64) (featureSelection, float64, bool) {
  ok := false
  // copy all features i with theta_{i+1} != 0
  t, c, b := obj.restoreNonzero(theta, features, kmers)
  if c < obj.N {
    ok = true
  }
  // compute gradient for selecting new features
  g := obj.gradient(data, t)
  i := make([]int, len(g))
  for k, _ := range i {
    i[k] = k
  }
  // sort gradient entries with respect to absolute values
  AbsFloatInt{g[1:], i[1:]}.SortReverse()
  // add new features
  for k := 1; k < len(i); k++ {
    if c >= obj.N {
      break
    }
    if !b[i[k]] {
      // feature was previously zero
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
  // set lambda to first gradient element not included in the feature set
  for k := 1; k < len(i); k++ {
    if !b[i[k]] {
      return math.Abs(g[k])
    }
  }
  return 0.0
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
  m := len(obj.featureSelector.Kmers)
  k := make([]int, len(obj.b))
  // remap data indices
  for i, j := 0, 0; j < len(obj.b); j++ {
    if obj.b[j] {
      k[j] =  i
      i   +=  1
    } else {
      k[j] = -1
    }
  }
  for i_ := 0; i_ < len(data); i_++ {
    i := []int    {}
    v := []float64{}
    for it := data[i_].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); obj.b[j] {
        i = append(i, k[j])
        v = append(v, it.GetValue())
      }
    }
    if obj.Cooccurrence {
      it1 := data[i_].ConstIterator()
      // skip first element
      it1.Next()
      for ; it1.Ok(); it1.Next() {
        it2 := it1.CloneConstIterator()
        it2.Next()
        for ; it2.Ok(); it2.Next() {
          i1 := it1.Index()-1
          i2 := it2.Index()-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          if obj.b[j] {
            i = append(i, k[j])
            v = append(v, it1.GetValue()*it2.GetValue())
          }
        }
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
