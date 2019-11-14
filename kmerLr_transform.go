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
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type TransformFull struct {
  Mu    []float64
  Sigma []float64
}

/* -------------------------------------------------------------------------- */

func (obj *TransformFull) Fit(config Config, data []ConstVector) {
  if len(data) == 0 {
    return
  }
  PrintStderr(config, 1, "Fitting data transform... ")
  n := len(data)
  m := data[0].Dim()
  if config.Cooccurrence {
    m = CoeffIndex(m-1).Dim() + 1
  }
  mu    := make([]float64, m)
  sigma := make([]float64, m)
  // compute mu
  for i := 0; i < n; i++ {
    it := data[i].ConstIterator()
    // skip first element
    it.Next()
    for ; it.Ok(); it.Next() {
      j := it.Index()
      mu[j] += it.GetValue()
    }
    if config.Cooccurrence {
      it1 := data[i].ConstIterator()
      // skip first element
      it1.Next()
      for ; it1.Ok(); it1.Next() {
        it2 := it1.CloneConstIterator()
        it2.Next()
        for ; it2.Ok(); it2.Next() {
          i1 := it1.Index()-1
          i2 := it2.Index()-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          mu[j] += it1.GetValue()*it2.GetValue()
        }
      }
    }
  }
  for j := 1; j < m; j++ {
    mu[j] /= float64(n)
  }
  k := make([]int, m)
  // compute sigma
  for i := 0; i < n; i++ {
    it := data[i].ConstIterator()
    // skip first element
    it.Next()
    for ; it.Ok(); it.Next() {
      j := it.Index()
      v := it.GetValue()
      k    [j] += 1
      sigma[j] += (v-mu[j])*(v-mu[j])
    }
    if config.Cooccurrence {
      it1 := data[i].ConstIterator()
      // skip first element
      it1.Next()
      for ; it1.Ok(); it1.Next() {
        it2 := it1.CloneConstIterator()
        it2.Next()
        for ; it2.Ok(); it2.Next() {
          i1 := it1.Index()-1
          i2 := it2.Index()-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          v  := it1.GetValue()*it2.GetValue()
          k    [j] += 1
          sigma[j] += (v-mu[j])*(v-mu[j])
        }
      }
    }
  }
  for j := 1; j < m; j++ {
    sigma[j] += float64(n-k[j])*mu[j]*mu[j]
  }
  for j := 1; j < m; j++ {
    if sigma[j] == 0.0 {
      sigma[j] = 1.0
    } else {
      sigma[j] = math.Sqrt(sigma[j]/float64(n))
    }
  }
  mu   [0]  = 0.0
  sigma[0]  = 1.0
  obj.Sigma = sigma
  obj.Mu    = mu
  PrintStderr(config, 1, "done\n")
}

func (obj TransformFull) Equals(t Transform) bool {
  if len(obj.Mu) != len(t.Mu) {
    return false
  }
  if len(obj.Sigma) != len(t.Sigma) {
    return false
  }
  for i := 0; i < len(obj.Mu); i++ {
    if math.Abs(obj.Mu[i] - t.Mu[i]) > 1e-12 {
      return false
    }
  }
  for i := 0; i < len(obj.Sigma); i++ {
    if math.Abs(obj.Sigma[i] - t.Sigma[i]) > 1e-12 {
      return false
    }
  }
  return true
}

func (obj TransformFull) Select(b []bool) Transform {
  tr := Transform{}
  if len(obj.Mu) > 0 {
    for i := 0; i < len(b); i++ {
      if b[i] {
        tr.Mu = append(tr.Mu, obj.Mu[i])
      }
    }
  }
  if len(obj.Sigma) > 0 {
    for i := 0; i < len(b); i++ {
      if b[i] {
        tr.Sigma = append(tr.Sigma, obj.Sigma[i])
      }
    }
  }
  return tr
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

type Transform struct {
  Mu    []float64
  Sigma []float64
}

/* -------------------------------------------------------------------------- */

func (t1 Transform) Equals(t2 Transform, f1, f2 FeatureIndices, k1, k2 KmerClassList) bool {
  // compare mu
  m := make(map[[2]KmerClassId]float64)
  for i, feature := range f1 {
    kmer1 := k1[feature[0]].KmerClassId
    kmer2 := k1[feature[1]].KmerClassId
    m[[2]KmerClassId{kmer1, kmer2}] = t1.Mu[i+1]
  }
  for i, feature := range f2 {
    kmer1 := k2[feature[0]].KmerClassId
    kmer2 := k2[feature[1]].KmerClassId
    if v, ok := m[[2]KmerClassId{kmer1, kmer2}]; ok {
      if math.Abs(v - t2.Mu[i+1]) > 1e-12 {
        return false
      }
    }
  }
  // compare sigma
  m = make(map[[2]KmerClassId]float64)
  for i, feature := range f1 {
    kmer1 := k1[feature[0]].KmerClassId
    kmer2 := k1[feature[1]].KmerClassId
    m[[2]KmerClassId{kmer1, kmer2}] = t1.Sigma[i+1]
  }
  for i, feature := range f2 {
    kmer1 := k2[feature[0]].KmerClassId
    kmer2 := k2[feature[1]].KmerClassId
    if v, ok := m[[2]KmerClassId{kmer1, kmer2}]; ok {
      if math.Abs(v - t2.Sigma[i+1]) > 1e-12 {
        return false
      }
    }
  }
  return true
}

func (obj Transform) Apply(config Config, data []ConstVector) {
  if len(obj.Mu) != len(obj.Sigma) {
    panic("internal error")
  }
  if len(obj.Mu) == 0 {
    return
  }
  PrintStderr(config, 1, "Normalizing data... ")
  n := len(data)
  m := data[0].Dim()
  for i := 0; i < n; i++ {
    if data[i].Dim() != m {
      panic("data has invalid dimension")
    }
    indices := data[i].(SparseConstRealVector).GetSparseIndices()
    values  := data[i].(SparseConstRealVector).GetSparseValues ()
    for j1, j2 := range indices {
      if j2 < len(obj.Mu) {
        values[j1] = (values[j1] - obj.Mu[j2])/obj.Sigma[j2]
      }
    }
    data[i] = UnsafeSparseConstRealVector(indices, values, m)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func (obj *Transform) ImportConfig(config ConfigDistribution, t ScalarType) error {

  mu   , ok := config.GetNamedParametersAsFloats("Mu"   ); if !ok {
    return fmt.Errorf("invalid config file")
  }
  sigma, ok := config.GetNamedParametersAsFloats("Sigma"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  obj.Mu    = mu
  obj.Sigma = sigma
  return nil
}

func (obj *Transform) ExportConfig() ConfigDistribution {
  return NewConfigDistribution("transform", obj)
}
