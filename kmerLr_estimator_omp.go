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

import . "github.com/pbenner/ngstat/estimation"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"
import . "github.com/pbenner/autodiff/logarithmetic"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type logisticRegression struct {
  Theta DenseBareRealVector
}

/* -------------------------------------------------------------------------- */

func (obj logisticRegression) Dim() int {
  return len(obj.Theta)-1
}

func (obj logisticRegression) LogPdf(v SparseConstRealVector) float64 {
  x     := v.GetSparseValues ()
  index := v.GetSparseIndices()
  // set r to first element of theta
  r := float64(obj.Theta[0])
  // loop over x
  i := 0
  n := len(index)
  // skip first element
  if index[i] == 0 {
    i++
  }
  for ; i < n; i++ {
    r += float64(x[i])*float64(obj.Theta[index[i]])
  }
  return -LogAdd(0.0, -r)
}

func (obj logisticRegression) gradient(data []ConstVector) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()
  w := 0.0
  g := make([]float64, m-1)

  for i := 0; i < n; i++ {
    r := obj.LogPdf(data[i].ConstSlice(0, m-1).(SparseConstRealVector))

    if data[i].ValueAt(m-1) == 1 {
      w = math.Exp(r) - 1.0
    } else {
      w = math.Exp(r)
    }
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 && j != m-1 {
        g[j] += w*it.GetConst().GetValue()
      }
    }
  }
  return g[1:]
}

/* -------------------------------------------------------------------------- */

type KmerLrOmpEstimator struct {
  vectorEstimator.LogisticRegression
  Kmers KmerClassList
  Hook  func(x ConstVector, change ConstScalar, epoch int) bool
}

/* -------------------------------------------------------------------------- */

func NewKmerLrOmpEstimator(config Config, kmers KmerClassList, hook HookType) *KmerLrOmpEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(kmers.Len()+1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    estimator.Hook          = hook
    estimator.Seed          = config.Seed
    estimator.L1Reg         = config.Lambda
    estimator.Epsilon       = config.Epsilon
    if config.MaxEpochs != 0 {
      estimator.MaxIterations = config.MaxEpochs
    }
    // alphabet parameters
    return &KmerLrOmpEstimator{*estimator, kmers, hook}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) Estimate(config Config, data []ConstVector) VectorPdf {
  data = obj.normalize(data)
  obj.bestFeatures(data)
  panic("exit")
  if err := EstimateOnSingleTrackConstData(config.SessionConfig, &obj.LogisticRegression, data); err != nil {
    log.Fatal(err)
  }
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{LogisticRegression: *r_.(*vectorDistribution.LogisticRegression)}
    r.KmerLrAlphabet.Binarize        = config.Binarize
    r.KmerLrAlphabet.KmerEquivalence = config.KmerEquivalence
    r.KmerLrAlphabet.Kmers           = obj   .Kmers
    return r.Sparsify()
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) bestFeatures(data []ConstVector) {
  r := logisticRegression{obj.Theta}
  g := r.gradient(data)
  k := make([]KmerClass, len(g))
  if len(g) != len(obj.Kmers) {
    panic("internal error")
  }
  for i := 0; i < len(g); i++ {
    g[i] = math.Abs(g[i])
    k[i] = obj.Kmers[i]
  }
  FloatKmer{g, k}.SortReverse()

  for i := 0; i < len(g); i++ {
    fmt.Printf("%e %v\n", g[i], k[i])
  }
}

func (obj *KmerLrOmpEstimator) normalize(data []ConstVector) []ConstVector {
  if len(data) == 0 {
    return data
  }
  n := len(data)
  m := data[0].Dim()
  x := make([]float64, m)
  for i := 0; i < n; i++ {
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 && j != m-1 {
        x[j] += it.GetConst().GetValue()*it.GetConst().GetValue()
      }
    }
  }
  for i := 0; i < n; i++ {
    indices := []int{}
    values  := []float64{}
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 && j != m-1 {
        indices = append(indices, j)
        values  = append(values , it.GetConst().GetValue()/x[j])
      } else {
        indices = append(indices, j)
        values  = append(values , it.GetConst().GetValue())
      }
    }
    data[i] = NewSparseConstRealVector(indices, values, m)
  }
  return data
}
