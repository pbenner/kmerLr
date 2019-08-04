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

func (obj logisticRegression) LogPdf(v SparseConstRealVector, gamma []float64) float64 {
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
    r += float64(x[i])/gamma[index[i]]*float64(obj.Theta[index[i]])
  }
  return -LogAdd(0.0, -r)
}

func (obj logisticRegression) gradient(data []ConstVector, gamma []float64) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()
  w := 0.0
  g := make([]float64, m-1)

  for i := 0; i < n; i++ {
    r := obj.LogPdf(data[i].ConstSlice(0, m-1).(SparseConstRealVector), gamma)

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
  // maximal number of active features
  n     int
  Hook  func(x ConstVector, change ConstScalar, epoch int) bool
}

/* -------------------------------------------------------------------------- */

func NewKmerLrOmpEstimator(config Config, kmers KmerClassList, n int, hook HookType) *KmerLrOmpEstimator {
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
    return &KmerLrOmpEstimator{*estimator, kmers, n, hook}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) Estimate(config Config, data []ConstVector) VectorPdf {
  gamma := obj.normalizationConstants(data)
  obj.selectFeatures(data, gamma)
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

func (obj *KmerLrOmpEstimator) selectData(data []ConstVector, k []int) []ConstVector {
  m := make([]bool, len(obj.Kmers))
  for _, j := range k {
    m[j] = true
  }
  r := make([]ConstVector, len(data))
  for i, _ := range data {
    values  := []float64{}
    indices := []int{}
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      j := it.Index()
      if m[j] {
        indices = append(indices, j)
        values  = append(values , it.GetConst().GetValue())
      }
    }
    r[i] = NewSparseBareRealVector(indices, values, len(k)+1)
  }
  return r
}

func (obj *KmerLrOmpEstimator) selectFeatures(data []ConstVector, gamma []float64) []int {
  r := logisticRegression{obj.Theta}
  g := r.gradient(data, gamma)
  k := make([]int, len(g))
  if len(g) != len(obj.Kmers) {
    panic("internal error")
  }
  for i := 0; i < len(g); i++ {
    g[i] = math.Abs(g[i])
    k[i] = i
  }
  FloatInt{g, k}.SortReverse()

  return k[0:obj.n]
}

func (obj *KmerLrOmpEstimator) normalizationConstants(data []ConstVector) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()
  x := make([]float64, m-1)
  for i := 0; i < n; i++ {
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 && j != m-1 {
        x[j] += it.GetConst().GetValue()*it.GetConst().GetValue()
      }
    }
  }
  return x
}
