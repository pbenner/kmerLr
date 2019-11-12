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
  DataTrain     []ConstVector
  DataTest      []ConstVector
  Labels        []bool
  Kmers           KmerClassList
  KmersMap        map[KmerClassId]int
  Cooccurrence    bool
  N               int
}

/* -------------------------------------------------------------------------- */

func newFeatureSelector(kmers KmerClassList, cooccurrence bool, data_train, data_test []ConstVector, labels []bool, class_weights [2]float64, n int) featureSelector {
  x_train := make([]ConstVector, len(data_train))
  x_test  := make([]ConstVector, len(data_test))
  for i, _ := range data_train {
    x_train[i] = data_train[i]
  }
  for i, _ := range data_test {
    x_test[i] = data_test[i]
  }
  m := make(map[KmerClassId]int)
  for i := 0; i < len(kmers); i++ {
    m[kmers[i].KmerClassId] = i
  }
  r := featureSelector{
    Kmers       : kmers,
    KmersMap    : m,
    DataTrain   : x_train,
    DataTest    : x_test,
    Labels      : labels,
    ClassWeights: class_weights,
    Cooccurrence: cooccurrence,
    N           : n }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) Select(theta []float64, features FeatureIndices, kmers KmerClassList, lambda float64) ([]float64, FeatureIndices, []ConstVector, []ConstVector, KmerClassList, float64, bool) {
  // copy all features i with theta_{i+1} != 0
  t, c, b := obj.restoreNonzero(theta, features, kmers)
  // compute gradient for selecting new features
  g := obj.gradient(t)
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
  t        = obj.selectTheta(b, t)
  x_train := obj.selectData (b, obj.DataTrain, c)
  x_test  := obj.selectData (b, obj.DataTest , c)
  k, f    := obj.selectKmers(b)
  l       := obj.computeLambda(b, g, i)
  return t, f, x_train, x_test, k, l, !features.Equals(f) || math.Abs(lambda - l) > 1e-6
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) computeLambda(b []bool, g []float64, i []int) float64 {
  // set lambda to first gradient element not included in the feature set
  for k := 1; k < len(i); k++ {
    if !b[i[k]] {
      return math.Abs(g[i[k]])/float64(len(obj.DataTrain))
    }
  }
  return 0.0
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) restoreNonzero(theta []float64, features FeatureIndices, kmers KmerClassList) ([]float64, int, []bool) {
  t := []float64(nil)
  b := []bool   (nil)
  c := 0
  m := obj.DataTrain[0].Dim()-1
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

func (obj featureSelector) selectTheta(b []bool, theta []float64) []float64 {
  r := []float64{}
  for i := 0; i < len(b); i++ {
    if b[i] {
      r = append(r, theta[i])
    }
  }
  return r
}

func (obj featureSelector) selectData(b []bool, data []ConstVector, n int) []ConstVector {
  m := obj.DataTrain[0].Dim()-1
  k := make([]int, len(b))
  x := make([]ConstVector, len(data))
  // remap data indices
  for i, j := 0, 0; j < len(b); j++ {
    if b[j] {
      k[j] =  i
      i   +=  1
    } else {
      k[j] = -1
    }
  }
  for i_ := 0; i_ < len(x); i_++ {
    i := []int    {}
    v := []float64{}
    for it := data[i_].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); b[j] {
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
          if b[j] {
            i = append(i, k[j])
            v = append(v, it1.GetValue()*it2.GetValue())
          }
        }
      }
    }
    // resize slice and restrict capacity
    i = append([]int    {}, i[0:len(i)]...)
    v = append([]float64{}, v[0:len(v)]...)
    x[i_] = UnsafeSparseConstRealVector(i, v, n+1)
  }
  return x
}

func (obj featureSelector) selectKmers(b []bool) (KmerClassList, FeatureIndices) {
  f := FeatureIndices{}
  m := obj.DataTrain[0].Dim()-1
  r := KmerClassList{}
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

func (obj featureSelector) gradient(theta []float64) []float64 {
  lr := logisticRegression{}
  lr.Theta        = theta
  lr.ClassWeights = obj.ClassWeights
  lr.Cooccurrence = obj.Cooccurrence
  return lr.Gradient(nil, obj.DataTrain, obj.Labels, nil)
}
