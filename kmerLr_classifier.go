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
import   "log"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLr struct {
  KmerLrFeatures
  Theta     []float64
  Transform   Transform
}

/* -------------------------------------------------------------------------- */

func NewKmerLr(theta []float64, alphabet KmerLrFeatures) *KmerLr {
  return &KmerLr{Theta: theta, KmerLrFeatures: alphabet}
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Clone() *KmerLr {
  r := KmerLr{}
  r.Theta = make([]float64, len(obj.Theta))
  for i := 0; i < len(obj.Theta); i++ {
    r.Theta[i] = obj.Theta[i]
  }
  r.KmerLrFeatures = obj.KmerLrFeatures.Clone()
  r.Transform      = obj.Transform     .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Predict(config Config, data []ConstVector) []float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.PoolLR
  r := make([]float64, len(data))
  for i, _ := range data {
    r[i] = lr.LogPdf(data[i].(SparseConstRealVector))
  }
  return r
}

func (obj *KmerLr) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.PoolLR
  if config.Balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  return lr.Loss(data, c)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Nonzero() int {
  n  := 0
  for i := 1; i < len(obj.Theta); i++ {
    if obj.Theta[i] != 0.0 {
      n++
    }
  }
  return n
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) GetKmerCounter() *KmerCounter {
  if counter, err := NewKmerCounter(obj.M, obj.N, obj.Complement, obj.Reverse, obj.Revcomp, obj.MaxAmbiguous, obj.Alphabet, obj.Kmers...); err != nil {
    log.Fatal(err)
    return nil
  } else {
    return counter
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) SelectData(config Config, data_ KmerDataSet) []ConstVector {
  data     := data_.Data
  data_dst := make([]ConstVector, len(data))
  kmers    := data_.Kmers
  kmap     := make(map[KmerClassId]int)
  for i, kmer := range kmers {
    kmap[kmer.KmerClassId] = i
  }
  for i_ := 0; i_ < len(data); i_++ {
    i := []int    {                 0 }
    v := []float64{data[i_].ValueAt(0)}
    for j, feature := range obj.Features {
      if feature[0] == feature[1] {
        i1, ok := kmap[obj.Kmers[feature[0]].KmerClassId]; if !ok {
          panic("internal error")
        }
        if value := data[i_].ValueAt(i1+1); value != 0.0 {
          i = append(i, j+1)
          v = append(v, value)
        }
      } else {
        i1, ok := kmap[obj.Kmers[feature[0]].KmerClassId]; if !ok {
          panic("internal error")
        }
        i2, ok := kmap[obj.Kmers[feature[1]].KmerClassId]; if !ok {
          panic("internal error")
        }
        if value := data[i_].ValueAt(i1+1)*data[i_].ValueAt(i2+1); value != 0.0 {
          i = append(i, j+1)
          v = append(v, value)
        }
      }
    }
    // resize slice and restrict capacity
    i = append([]int    {}, i[0:len(i)]...)
    v = append([]float64{}, v[0:len(v)]...)
    data_dst[i_] = UnsafeSparseConstRealVector(i, v, len(obj.Features)+1)
  }
  obj.Transform.Apply(config, data_dst)
  return data_dst
}
