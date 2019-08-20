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

/* -------------------------------------------------------------------------- */

type Transform struct {
  Mu    []float64
  Sigma []float64
}

/* -------------------------------------------------------------------------- */

func (obj *Transform) TransformFit(data []ConstVector) {
  if len(data) == 0 {
    return
  }
  n := len(data)
  m := data[0].Dim()
  mu    := make([]float64, m)
  sigma := make([]float64, m)
  // compute mu
  for i := 0; i < n; i++ {
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      mu[it.Index()] += it.GetConst().GetValue()
    }
  }
  for j := 1; j < m-1; j++ {
    mu[j] /= float64(n)
  }
  mu[0  ] = 0.0
  mu[m-1] = 0.0
  k := make([]int, m)
  // compute sigma
  for i := 0; i < n; i++ {
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      j := it.Index()
      v := it.GetConst().GetValue()
      k    [j] += 1
      sigma[j] += (v-mu[j])*(v-mu[j])
    }
  }
  for j := 1; j < m-1; j++ {
    sigma[j] += float64(n-k[j])*mu[j]*mu[j]
  }
  for j := 1; j < m-1; j++ {
    if sigma[j] == 0.0 {
      sigma[j] = 1.0
    } else {
      sigma[j] = math.Sqrt(sigma[j]/float64(n))
    }
  }
  sigma[0  ] = 1.0
  sigma[m-1] = 1.0
  obj.Sigma = sigma
  obj.Mu    = mu
}

func (obj Transform) TransformApply(data []ConstVector) []ConstVector {
  if len(obj.Mu) != len(obj.Sigma) {
    panic("internal error")
  }
  if len(obj.Mu) == 0 {
    return data
  }
  n := len(data)
  m := data[0].Dim()
  for i := 0; i < n; i++ {
    if data[i].Dim() != m {
      panic("data has invalid dimension")
    }
    indices := data[i].(SparseConstRealVector).GetSparseIndices()
    values  := data[i].(SparseConstRealVector).GetSparseValues ()
    for j1, j2 := range indices {
      values[j1] = (values[j1] - ConstReal(obj.Mu[j2]))/ConstReal(obj.Sigma[j2])
    }
    data[i] = UnsafeSparseConstRealVector(indices, values, m)
  }
  return data
}

func (obj Transform) TransformEquals(t Transform) bool {
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
