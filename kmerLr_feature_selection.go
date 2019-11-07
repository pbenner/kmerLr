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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type featureSelector struct {
  ClassWeights [2]float64
  // static list of available kmers
  Kmers    KmerClassList
  // static data set
  Data   []ConstVector
  Labels []bool
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) Select(theta []float64, n int, cooccurrence bool) (FeatureIndices, []ConstVector, KmerClassList, float64) {
  f := FeatureIndices{}
  l := 0.0
  b := make([]bool, len(theta))
  b[0] = true
  // count non-zero entries in theta
  for k := 1; k < len(theta); k++ {
    if theta[k] != 0.0 {
      f    = append(f, [2]int{k-1, k-1})
      b[k] = true
    }
  }
  if len(f) < n {
    g := obj.gradient(theta, n, cooccurrence)
    i := make([]int, len(g))
    m := obj.Data[0].Dim()-1
    for k, _ := range i {
      i[k] = k
    }
    AbsFloatInt{g[1:], i[1:]}.SortReverse()
    // add new features
    for k := 1; k < len(i); k++ {
      if len(f) >= n {
        break
      }
      if j := i[k]; theta[j] == 0.0 {
        // feature was previously zero
        i1, i2 := CoeffIndex(m).Sub2Ind(j-1)
        f    = append(f, [2]int{i1, i2})
        b[k] = true
      }
      // new lambda value
      l = g[k]
    }
  }
  x := obj.selectData (b, cooccurrence)
  k := obj.selectKmers(b)
  return f, x, k, l
}

/* -------------------------------------------------------------------------- */

func (obj featureSelector) selectData(b []bool, cooccurrence bool) []ConstVector {
  x := make([]ConstVector, len(obj.Data))
  m := obj.Data[0].Dim()-1
  for i_ := 0; i_ < len(x); i_++ {
    i := []int    {}
    v := []float64{}
    for it := obj.Data[i_].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); b[j] {
        i = append(i, j)
        v = append(v, it.GetValue())
      }
    }
    if cooccurrence {
      for it1 := obj.Data[i_].ConstIterator(); it1.Ok(); it1.Next() {
        it2 := it1.CloneConstIterator()
        it2.Next()
        for ; it2.Ok(); it2.Next() {
          i1 := it1.Index()-1
          i2 := it2.Index()-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          i   = append(i, j)
          v   = append(v, it1.GetValue()*it2.GetValue())
        }
      }
    }
    // resize slice and restrict capacity
    i = append([]int    {}, i[0:len(i)]...)
    v = append([]float64{}, v[0:len(v)]...)
    x[i_] = UnsafeSparseConstRealVector(i, v, m+1)
  }
  return x
}

func (obj featureSelector) selectKmers(b []bool) KmerClassList {
  m := obj.Data[0].Dim()-1
  r := KmerClassList{}
  for i := 0; i < m; i++ {
    if b[i+1] {
      r = append(r, obj.Kmers[i])
    }
  }
  return r
}

func (obj featureSelector) gradient(theta []float64, n int, cooccurrence bool) []float64 {
  lr := logisticRegression{}
  lr.Theta        = theta
  lr.ClassWeights = obj.ClassWeights
  lr.Cooccurrence = cooccurrence
  return lr.Gradient(nil, obj.Data, obj.Labels, nil)
}
