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
//import   "log"

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

func (obj featureSelector) Select(theta []float64, n int, cooccurrence bool) {
  f := FeatureIndices{}
  // count non-zero entries in theta
  for k, _ := range theta[1:] {
    if theta[k] != 0.0 {
      f = append(f, [2]int{k, k})
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
        i1, i2 := CoeffIndex(m).Sub2Ind(j)
        f = append(f, [2]int{i1, i2})
      }
    }
  }
}

func (obj featureSelector) gradient(theta []float64, n int, cooccurrence bool) []float64 {
  lr := logisticRegression{}
  lr.Theta        = theta
  lr.ClassWeights = obj.ClassWeights
  lr.Cooccurrence = cooccurrence
  return lr.Gradient(nil, obj.Data, obj.Labels, nil)
}
