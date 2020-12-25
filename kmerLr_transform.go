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
import   "log"
import   "math"
import   "strings"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type TransformFull struct {
  Offset []float64
  Scale  []float64
}

/* -------------------------------------------------------------------------- */

func (obj *TransformFull) Fit(config Config, data []ConstVector, cooccurrence bool) {
  switch strings.ToLower(config.DataTransform) {
  case "":
  case "none":
  case "standardize":
    PrintStderr(config, 1, "WARNING: Standardizing data will result in dense data and slow down computations!")
    obj.FitStandardizer(config, data, cooccurrence)
  case "variance-scaler":
    obj.FitVarianceScaler(config, data, cooccurrence)
  case "max-abs-scaler":
    obj.FitMaxAbsScaler(config, data, cooccurrence)
  case "mean-scaler":
    obj.FitMeanScaler(config, data, cooccurrence)
  default:
    log.Fatal("invalid data transform")
    panic("internal error")
  }
}

func (obj *TransformFull) fitStandardizer(config Config, data []ConstVector, cooccurrence bool) {
  if len(data) == 0 {
    return
  }
  n := len(data)
  m := data[0].Dim()-1
  offset := []float64{}
  scale  := []float64{}
  if cooccurrence {
    offset = make([]float64, CoeffIndex(m).Dim()+1)
    scale  = make([]float64, CoeffIndex(m).Dim()+1)
  } else {
    offset = make([]float64, m+1)
    scale  = make([]float64, m+1)
  }
  // compute offset
  for i_ := 0; i_ < n; i_++ {
    if data[i_].Dim() != m+1 {
      panic("internal error")
    }
    i := data[i_].(SparseConstFloat64Vector).GetSparseIndices()
    v := data[i_].(SparseConstFloat64Vector).GetSparseValues ()
    q := len(i)

    for j := 1; j < q; j++ {
      offset[i[j]] += v[j]
    }
    if cooccurrence {
      config.Pool.RangeJob(1, q, func(j1 int, pool threadpool.ThreadPool, erf func() error) error {
        for j2 := j1+1; j2 < q; j2++ {
          i1    := i[j1]-1
          i2    := i[j2]-1
          j     := CoeffIndex(m).Ind2Sub(i1, i2)
          offset[j] += v[j1]*v[j2]
        }
        return nil
      })
    }
  }
  // normalize mean (probably less floating point operations if normalization is here)
  for j := 1; j < len(offset); j++ {
    offset[j] /= float64(n)
  }
  k := make([]int, len(offset))
  // compute scale
  for i_ := 0; i_ < n; i_++ {
    if data[i_].Dim() != m+1 {
      panic("internal error")
    }
    i := data[i_].(SparseConstFloat64Vector).GetSparseIndices()
    v := data[i_].(SparseConstFloat64Vector).GetSparseValues ()
    q := len(i)

    for j := 1; j < q; j++ {
      j_        := i[j]
      k    [j_] += 1
      scale[j_] += (v[j]-offset[j_])*(v[j]-offset[j_])
    }
    if cooccurrence {
      config.Pool.RangeJob(1, q, func(j1 int, pool threadpool.ThreadPool, erf func() error) error {
        for j2 := j1+1; j2 < q; j2++ {
          i1 := i[j1]-1
          i2 := i[j2]-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          k    [j] += 1
          scale[j] += (v[j1]*v[j2]-offset[j])*(v[j1]*v[j2]-offset[j])
        }
        return nil
      })
    }
  }
  for j := 1; j < len(scale); j++ {
    // account for zero entries
    sj := scale[j] + float64(n-k[j])*offset[j]*offset[j]
    // compute standard deviation
    if sj == 0.0 {
      scale[j] = 1.0
    } else {
      scale[j] = math.Sqrt(sj/float64(n-1))
    }
    // invert
    scale[j] = 1.0/scale[j]
  }
  offset[0]  = 0.0
  scale [0]  = 1.0
  obj.Offset = offset
  obj.Scale  = scale
}

func (obj *TransformFull) FitStandardizer(config Config, data []ConstVector, cooccurrence bool) {
  PrintStderr(config, 1, "Fitting data transform (standardizer)... ")
  obj.fitStandardizer(config, data, cooccurrence)
  PrintStderr(config, 1, "done\n")
}

func (obj *TransformFull) FitVarianceScaler(config Config, data []ConstVector, cooccurrence bool) {
  PrintStderr(config, 1, "Fitting data transform (variance scaler)... ")
  obj.fitStandardizer(config, data, cooccurrence)
  obj.Offset = nil
  PrintStderr(config, 1, "done\n")
}

func (obj *TransformFull) FitMaxAbsScaler(config Config, data []ConstVector, cooccurrence bool) {
  if len(data) == 0 {
    return
  }
  PrintStderr(config, 1, "Fitting data transform (max-abs scaler)... ")
  n := len(data)
  m := data[0].Dim()-1
  scale := []float64{}
  if cooccurrence {
    scale = make([]float64, CoeffIndex(m).Dim()+1)
  } else {
    scale = make([]float64, m+1)
  }
  // compute offset
  for i_ := 0; i_ < n; i_++ {
    if data[i_].Dim() != m+1 {
      panic("internal error")
    }
    i := data[i_].(SparseConstFloat64Vector).GetSparseIndices()
    v := data[i_].(SparseConstFloat64Vector).GetSparseValues ()
    q := len(i)

    for j := 1; j < q; j++ {
      scale[i[j]] = math.Max(scale[i[j]], math.Abs(v[j]))
    }
    if cooccurrence {
      config.Pool.RangeJob(1, q, func(j1 int, pool threadpool.ThreadPool, erf func() error) error {
        for j2 := j1+1; j2 < q; j2++ {
          i1    := i[j1]-1
          i2    := i[j2]-1
          j     := CoeffIndex(m).Ind2Sub(i1, i2)
          scale[j] = math.Max(scale[j], math.Abs(v[j1]*v[j2]))
        }
        return nil
      })
    }
  }
  scale[0]   = 1.0
  obj.Offset = nil
  obj.Scale  = scale
  PrintStderr(config, 1, "done\n")
}

func (obj *TransformFull) FitMeanScaler(config Config, data []ConstVector, cooccurrence bool) {
  if len(data) == 0 {
    return
  }
  PrintStderr(config, 1, "Fitting data transform (mean scaler)... ")
  n := len(data)
  m := data[0].Dim()-1
  scale := []float64{}
  if cooccurrence {
    scale = make([]float64, CoeffIndex(m).Dim()+1)
  } else {
    scale = make([]float64, m+1)
  }
  // compute offset
  for i_ := 0; i_ < n; i_++ {
    if data[i_].Dim() != m+1 {
      panic("internal error")
    }
    i := data[i_].(SparseConstFloat64Vector).GetSparseIndices()
    v := data[i_].(SparseConstFloat64Vector).GetSparseValues ()
    q := len(i)

    for j := 1; j < q; j++ {
      scale[i[j]] += v[j]
    }
    if cooccurrence {
      config.Pool.RangeJob(1, q, func(j1 int, pool threadpool.ThreadPool, erf func() error) error {
        for j2 := j1+1; j2 < q; j2++ {
          i1    := i[j1]-1
          i2    := i[j2]-1
          j     := CoeffIndex(m).Ind2Sub(i1, i2)
          scale[j] += v[j1]*v[j2]
        }
        return nil
      })
    }
  }
  // normalize mean (probably less floating point operations if normalization is here)
  for j := 1; j < len(scale); j++ {
    scale[j] /= float64(n)
  }
  scale[0]   = 1.0
  obj.Offset = nil
  obj.Scale  = scale
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func (obj TransformFull) Nil() bool {
  return len(obj.Offset) == 0 && len(obj.Scale) == 0
}

func (obj TransformFull) Apply(value float64, j int) float64 {
  if len(obj.Offset) > 0 {
    value = value - obj.Offset[j]
  }
  if len(obj.Scale ) > 0 {
    value = value * obj.Scale[j]
  }
  return value
}

func (obj TransformFull) Equals(t Transform) bool {
  if len(obj.Offset) != len(t.Offset) {
    return false
  }
  if len(obj.Scale ) != len(t.Scale ) {
    return false
  }
  for i := 0; i < len(obj.Offset); i++ {
    if math.Abs(obj.Offset[i] - t.Offset[i]) > 1e-12 {
      return false
    }
  }
  for i := 0; i < len(obj.Scale); i++ {
    if math.Abs(obj.Scale[i] - t.Scale[i]) > 1e-12 {
      return false
    }
  }
  return true
}

func (obj TransformFull) Select(b []bool) Transform {
  tr := Transform{}
  if len(obj.Offset) > 0 {
    for i := 0; i < len(b); i++ {
      if b[i] {
        tr.Offset = append(tr.Offset, obj.Offset[i])
      }
    }
  }
  if len(obj.Scale) > 0 {
    for i := 0; i < len(b); i++ {
      if b[i] {
        tr.Scale = append(tr.Scale, obj.Scale[i])
      }
    }
  }
  return tr
}

func (obj TransformFull) SelectAll() Transform {
  tr := Transform{}
  tr.Offset = obj.Offset
  tr.Scale  = obj.Scale
  return tr
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

type Transform struct {
  Offset []float64
  Scale  []float64
}

/* -------------------------------------------------------------------------- */

func NewTransform(n int, with_offset, with_scale bool) Transform {
  if n < 0 {
    panic("internal error")
  }
  if !with_offset && !with_scale {
    panic("internal error")
  }
  t := Transform{}
  if with_offset {
    t.Offset    = make([]float64, n+1)
    t.Offset[0] = 0.0
  }
  if with_scale  {
    t.Scale     = make([]float64, n+1)
    t.Scale[0]  = 1.0
  }
  return t
}

func (obj Transform) Clone() Transform {
  t := Transform{}
  t.Offset = make([]float64, len(obj.Offset))
  t.Scale  = make([]float64, len(obj.Scale ))
  copy(t.Offset, obj.Offset)
  copy(t.Scale , obj.Scale )
  return t
}

func (obj Transform) Nil() bool {
  return len(obj.Offset) == 0 && len(obj.Scale) == 0
}

func (obj Transform) Dim() int {
  n := len(obj.Offset)
  m := len(obj.Scale )
  if n != 0 {
    return n
  } else {
    return m
  }
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
      if len(t2.Offset) > 0 {
        if t1.Offset[i+1] == 0.0 {
          t1.Offset[i+1] = t2.Offset[j+1]
        } else
        if t2.Offset[j+1] != 0.0 {
          if math.Abs(t1.Offset[i+1] - t2.Offset[j+1]) > 1e-12 {
            return fmt.Errorf("joining transforms failed: transforms are incompatible")
          }
        }
      }
      if len(t2.Scale) > 0 {
        if t1.Scale[i+1] == 0.0 {
          t1.Scale[i+1] = t2.Scale[j+1]
        } else
        if t2.Scale[j+1] != 0.0 {
          if math.Abs(t1.Scale[i+1] - t2.Scale[j+1]) > 1e-12 {
            return fmt.Errorf("joining transforms failed: transforms are incompatible")
          }
        }
      }
    }
  }
  return nil
}

func (t1 Transform) InsertScores(t2 Transform, f1, f2 FeatureIndices, i1, i2 []int) error {
  if t1.Dim() != len(f1)+1 {
    panic("internal error")
  }
  if t2.Dim() != len(f2)+1 {
    panic("internal error")
  }
  // create index
  m := make(map[[2]int]int)
  for i, feature := range f1 {
    m[[2]int{i1[feature[0]], i1[feature[1]]}] = i
  }
  for j, feature := range f2 {
    // insert only if feature is present in target transform
    if i, ok := m[[2]int{i2[feature[0]], i2[feature[1]]}]; ok {
      if t1.Offset[i+1] == 0.0 {
        t1.Offset[i+1] = t2.Offset[j+1]
      } else
      if t2.Offset[j+1] != 0.0 {
        if math.Abs(t1.Offset[i+1] - t2.Offset[j+1]) > 1e-12 {
          return fmt.Errorf("joining transforms failed: transforms are incompatible")
        }
      }
      if t1.Scale[i+1] == 0.0 {
        t1.Scale[i+1] = t2.Scale[j+1]
      } else
      if t2.Scale[j+1] != 0.0 {
        if math.Abs(t1.Scale[i+1] - t2.Scale[j+1]) > 1e-12 {
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
  // compare offset
  m := make(map[[2]KmerClassId]float64)
  for i, feature := range f1 {
    kmer1 := k1[feature[0]].KmerClassId
    kmer2 := k1[feature[1]].KmerClassId
    m[[2]KmerClassId{kmer1, kmer2}] = t1.Offset[i+1]
  }
  for i, feature := range f2 {
    kmer1 := k2[feature[0]].KmerClassId
    kmer2 := k2[feature[1]].KmerClassId
    if v, ok := m[[2]KmerClassId{kmer1, kmer2}]; ok {
      if math.Abs(v - t2.Offset[i+1]) > 1e-12 {
        return false
      }
    }
  }
  // compare scale
  m = make(map[[2]KmerClassId]float64)
  for i, feature := range f1 {
    kmer1 := k1[feature[0]].KmerClassId
    kmer2 := k1[feature[1]].KmerClassId
    m[[2]KmerClassId{kmer1, kmer2}] = t1.Scale[i+1]
  }
  for i, feature := range f2 {
    kmer1 := k2[feature[0]].KmerClassId
    kmer2 := k2[feature[1]].KmerClassId
    if v, ok := m[[2]KmerClassId{kmer1, kmer2}]; ok {
      if math.Abs(v - t2.Scale[i+1]) > 1e-12 {
        return false
      }
    }
  }
  return true
}

func (t1 Transform) EqualsScores(t2 Transform, f1, f2 FeatureIndices, i1, i2 []int) bool {
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
  // compare offset
  m := make(map[[2]int]float64)
  for i, feature := range f1 {
    m[[2]int{i1[feature[0]], i1[feature[1]]}] = t1.Offset[i+1]
  }
  for i, feature := range f2 {
    if v, ok := m[[2]int{i2[feature[0]], i2[feature[1]]}]; ok {
      if math.Abs(v - t2.Offset[i+1]) > 1e-12 {
        return false
      }
    }
  }
  // compare scale
  m = make(map[[2]int]float64)
  for i, feature := range f1 {
    m[[2]int{i1[feature[0]], i1[feature[1]]}] = t1.Scale[i+1]
  }
  for i, feature := range f2 {
    if v, ok := m[[2]int{i2[feature[0]], i2[feature[1]]}]; ok {
      if math.Abs(v - t2.Scale[i+1]) > 1e-12 {
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
  if len(obj.Offset) == 0 && len(obj.Scale) == 0 {
    return
  }
  PrintStderr(config, 1, "Normalizing data... ")
  n := len(data)
  m := data[0].Dim()
  for i := 0; i < n; i++ {
    if data[i].Dim() != m {
      panic("data has invalid dimension")
    }
    indices := data[i].(SparseConstFloat64Vector).GetSparseIndices()
    values  := data[i].(SparseConstFloat64Vector).GetSparseValues ()
    if len(obj.Offset) > 0 && len(obj.Scale) > 0 {
      result := make([]float64, m)
      for k, j := 0, 0; k < m; k++ {
        if j < len(values) && indices[j] == k {
          result[k] = (values[j] - obj.Offset[k])*obj.Scale[k]; j++
        } else {
          result[k] = (      0.0 - obj.Offset[k])*obj.Scale[k]
        }
      }
      data[i] = AsSparseConstFloat64Vector(NewDenseFloat64Vector(result))
    } else
    if len(obj.Offset) > 0 {
      result := make([]float64, m)
      for k, j := 0, 0; k < m; k++ {
        if j < len(values) && indices[j] == k {
          result[k] = (values[j] - obj.Offset[k]); j++
        } else {
          result[k] = (      0.0 - obj.Offset[k])
        }
      }
      data[i] = AsSparseConstFloat64Vector(NewDenseFloat64Vector(result))
    } else {
      for j1, j2 := range indices {
        values[j1] = values[j1]*obj.Scale[j2]
      }
      data[i] = UnsafeSparseConstFloat64Vector(indices, values, m)
    }
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func (obj *Transform) ImportConfig(config ConfigDistribution, t ScalarType) error {

  offset, ok := config.GetNamedParametersAsFloats("Offset"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  scale , ok := config.GetNamedParametersAsFloats("Scale" ); if !ok {
    return fmt.Errorf("invalid config file")
  }
  obj.Offset = offset
  obj.Scale  = scale
  return nil
}

func (obj *Transform) ExportConfig() ConfigDistribution {
  return NewConfigDistribution("transform", obj)
}
