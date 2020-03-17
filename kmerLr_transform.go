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
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type TransformFull struct {
  Mu    []float64
  Sigma []float64
}

/* -------------------------------------------------------------------------- */

func (obj *TransformFull) Fit(config Config, data []ConstVector, cooccurrence bool) {
  if len(data) == 0 {
    return
  }
  PrintStderr(config, 1, "Fitting data transform... ")
  n := len(data)
  m := data[0].Dim()-1
  mu    := []float64{}
  sigma := []float64{}
  if cooccurrence {
    mu    = make([]float64, CoeffIndex(m).Dim()+1)
    sigma = make([]float64, CoeffIndex(m).Dim()+1)
  } else {
    mu    = make([]float64, m+1)
    sigma = make([]float64, m+1)
  }
  // compute mu
  for i_ := 0; i_ < n; i_++ {
    if data[i_].Dim() != m+1 {
      panic("internal error")
    }
    i := data[i_].(SparseConstRealVector).GetSparseIndices()
    v := data[i_].(SparseConstRealVector).GetSparseValues ()
    q := len(i)

    for j := 1; j < q; j++ {
      mu[i[j]] += v[j]
    }
    if cooccurrence {
      config.Pool.RangeJob(1, q, func(j1 int, pool threadpool.ThreadPool, erf func() error) error {
        for j2 := j1+1; j2 < q; j2++ {
          i1    := i[j1]-1
          i2    := i[j2]-1
          j     := CoeffIndex(m).Ind2Sub(i1, i2)
          mu[j] += v[j1]*v[j2]
        }
        return nil
      })
    }
  }
  // normalize mean (probably less floating point operations if normalization is here)
  for j := 1; j < len(mu); j++ {
    mu[j] /= float64(n)
  }
  k := make([]int, len(mu))
  // compute sigma
  for i_ := 0; i_ < n; i_++ {
    if data[i_].Dim() != m+1 {
      panic("internal error")
    }
    i := data[i_].(SparseConstRealVector).GetSparseIndices()
    v := data[i_].(SparseConstRealVector).GetSparseValues ()
    q := len(i)

    for j := 1; j < q; j++ {
      j_        := i[j]
      k    [j_] += 1
      sigma[j_] += (v[j]-mu[j_])*(v[j]-mu[j_])
    }
    if cooccurrence {
      config.Pool.RangeJob(1, q, func(j1 int, pool threadpool.ThreadPool, erf func() error) error {
        for j2 := j1+1; j2 < q; j2++ {
          i1 := i[j1]-1
          i2 := i[j2]-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          k    [j] += 1
          sigma[j] += (v[j1]*v[j2]-mu[j])*(v[j1]*v[j2]-mu[j])
        }
        return nil
      })
    }
  }
  for j := 1; j < len(sigma); j++ {
    // account for zero entries
    sj := sigma[j] + float64(n-k[j])*mu[j]*mu[j]
    // compute standard deviation
    if sj == 0.0 {
      sigma[j] = 1.0
    } else {
      sigma[j] = math.Sqrt(sj/float64(n))
    }
  }
  mu   [0]  = 0.0
  sigma[0]  = 1.0
  obj.Sigma = sigma
  obj.Mu    = mu
  PrintStderr(config, 1, "done\n")
}

func (obj TransformFull) Nil() bool {
  return len(obj.Mu) == 0
}

func (obj TransformFull) Apply(value float64, j int) float64 {
  return (value - obj.Mu[j])/obj.Sigma[j]
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

func NewTransform(n int) Transform {
  t := Transform{}
  t.Mu    = make([]float64, n)
  t.Sigma = make([]float64, n)
  return t
}

func (obj Transform) Clone() Transform {
  t := Transform{}
  t.Mu    = make([]float64, len(obj.Mu))
  t.Sigma = make([]float64, len(obj.Sigma))
  copy(t.Mu   , obj.Mu)
  copy(t.Sigma, obj.Sigma)
  return t
}

func (obj Transform) Nil() bool {
  return len(obj.Mu) == 0
}

func (obj Transform) Dim() int {
  return len(obj.Mu)
}

func (t1 Transform) Insert(t2 Transform, f1, f2 FeatureIndices, k1, k2 KmerClassList) error {
  if t1.Dim() != len(f1)+1 {
    panic("internal error")
  }
  if t2.Dim() != len(f2)+1 {
    panic("internal error")
  }
  // create index
  m := make(map[[2]KmerClassId]int)
  for i, feature := range f1 {
    kmer1 := k1[feature[0]].KmerClassId
    kmer2 := k1[feature[1]].KmerClassId
    m[[2]KmerClassId{kmer1, kmer2}] = i
  }
  for j, feature := range f2 {
    kmer1 := k2[feature[0]].KmerClassId
    kmer2 := k2[feature[1]].KmerClassId
    // insert only if feature is present in target transform
    if i, ok := m[[2]KmerClassId{kmer1, kmer2}]; ok {
      if t1.Mu[i+1] == 0.0 {
        t1.Mu[i+1] = t2.Mu[j+1]
      } else
      if t2.Mu[j+1] != 0.0 {
        if math.Abs(t1.Mu[i+1] - t2.Mu[j+1]) > 1e-12 {
          return fmt.Errorf("joining transforms failed: transforms are incompatible")
        }
      }
      if t1.Sigma[i+1] == 0.0 {
        t1.Sigma[i+1] = t2.Sigma[j+1]
      } else
      if t2.Sigma[j+1] != 0.0 {
        if math.Abs(t1.Sigma[i+1] - t2.Sigma[j+1]) > 1e-12 {
          return fmt.Errorf("joining transforms failed: transforms are incompatible")
        }
      }
    }
  }
  return nil
}

func (t1 Transform) InsertScores(t2 Transform, f1, f2 FeatureIndices) error {
  if t1.Dim() != len(f1)+1 {
    panic("internal error")
  }
  if t2.Dim() != len(f2)+1 {
    panic("internal error")
  }
  // create index
  m := make(map[[2]int]int)
  for i, feature := range f1 {
    m[feature] = i
  }
  for j, feature := range f2 {
    // insert only if feature is present in target transform
    if i, ok := m[feature]; ok {
      if t1.Mu[i+1] == 0.0 {
        t1.Mu[i+1] = t2.Mu[j+1]
      } else
      if t2.Mu[j+1] != 0.0 {
        if math.Abs(t1.Mu[i+1] - t2.Mu[j+1]) > 1e-12 {
          return fmt.Errorf("joining transforms failed: transforms are incompatible")
        }
      }
      if t1.Sigma[i+1] == 0.0 {
        t1.Sigma[i+1] = t2.Sigma[j+1]
      } else
      if t2.Sigma[j+1] != 0.0 {
        if math.Abs(t1.Sigma[i+1] - t2.Sigma[j+1]) > 1e-12 {
          return fmt.Errorf("joining transforms failed: transforms are incompatible")
        }
      }
    }
  }
  return nil
}

func (t1 Transform) Equals(t2 Transform, f1, f2 FeatureIndices, k1, k2 KmerClassList) bool {
  { // check if transforms are nil
    b1 := false
    b2 := false
    if t1.Nil() && len(f1) != 0 {
      b1 = true
    }
    if t2.Nil() && len(f2) != 0 {
      b2 = true
    }
    if b1 != b2 {
      return false
    }
    if b1 == true && b2 == true {
      return true
    }
  }
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

func (t1 Transform) EqualsScores(t2 Transform, f1, f2 FeatureIndices) bool {
  { // check if transforms are nil
    b1 := false
    b2 := false
    if t1.Nil() && len(f1) != 0 {
      b1 = true
    }
    if t2.Nil() && len(f2) != 0 {
      b2 = true
    }
    if b1 != b2 {
      return false
    }
    if b1 == true && b2 == true {
      return true
    }
  }
  // compare mu
  m := make(map[[2]int]float64)
  for i, feature := range f1 {
    m[[2]int{feature[0], feature[1]}] = t1.Mu[i+1]
  }
  for i, feature := range f2 {
    if v, ok := m[[2]int{feature[0], feature[1]}]; ok {
      if math.Abs(v - t2.Mu[i+1]) > 1e-12 {
        return false
      }
    }
  }
  // compare sigma
  m = make(map[[2]int]float64)
  for i, feature := range f1 {
    m[[2]int{feature[0], feature[1]}] = t1.Sigma[i+1]
  }
  for i, feature := range f2 {
    if v, ok := m[[2]int{feature[0], feature[1]}]; ok {
      if math.Abs(v - t2.Sigma[i+1]) > 1e-12 {
        return false
      }
    }
  }
  return true
}

func (obj Transform) Apply(config Config, data []ConstVector) {
  if len(data) == 0 {
    return
  }
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
