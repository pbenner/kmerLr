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
import   "sort"

import . "github.com/pbenner/ngstat/estimation"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"
import . "github.com/pbenner/autodiff/logarithmetic"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type logisticRegression struct {
  Theta []float64
}

/* -------------------------------------------------------------------------- */

func (obj logisticRegression) Dim() int {
  return len(obj.Theta)-1
}

func (obj logisticRegression) LogPdf(v SparseConstRealVector, gamma []float64) float64 {
  x     := v.GetSparseValues ()
  index := v.GetSparseIndices()
  // set r to first element of theta
  r := obj.Theta[0]
  // loop over x
  i := 0
  n := len(index)
  // skip first element
  if index[i] == 0 {
    i++
  }
  for ; i < n; i++ {
    r += float64(x[i])/gamma[index[i]]*obj.Theta[index[i]]
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
  // list of all features
  Kmers KmerClassList
  // full coefficients vector
  theta_ []float64
  // set of active features
  active []int
  // maximal number of active features
  n    int
  Hook func(x ConstVector, change ConstScalar, epoch int) bool
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
    r := KmerLrOmpEstimator{}
    r.LogisticRegression = *estimator
    r.Kmers  = kmers
    r.theta_ = make([]float64, kmers.Len()+1)
    r.n      = n
    r.Hook   = hook
    return &r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) Estimate(config Config, data []ConstVector) VectorPdf {
  gamma := obj.normalizationConstants(data)
  for {
    // select a subset of features using OMP
    features, ok := obj.selectFeatures(data, gamma); if !ok {
      break
    }
    // get subset of data and coefficients for these features
    features_data := obj.selectData        (data, features)
    features_coef := obj.selectCoefficients(data, features)
    // copy coefficients to estimator
    if err := obj.LogisticRegression.SetParameters(features_coef); err != nil {
      log.Fatal(err)
    }
    // estimate reduced set of coefficients
    if err := EstimateOnSingleTrackConstData(config.SessionConfig, &obj.LogisticRegression, features_data); err != nil {
      log.Fatal(err)
    }
    // copy coefficients to backup vector
    obj.saveCoefficients(features)
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

func (obj *KmerLrOmpEstimator) saveCoefficients(k []int) {
  for j, _ := range obj.theta_ {
    obj.theta_[j] = 0.0
  }
  for i, j := range k {
    obj.theta_[j] = obj.Theta.ValueAt(i+1)
  }
  obj.theta_[0] = obj.Theta.ValueAt(0)
}

func (obj *KmerLrOmpEstimator) selectCoefficients(data []ConstVector, k []int) Vector {
  v := make([]float64, len(k)+1)
  for i, j := range k {
    v[i+1] = obj.theta_[j]
  }
  v[0] = obj.theta_[0]
  return NewDenseBareRealVector(v)
}

func (obj *KmerLrOmpEstimator) selectData(data []ConstVector, k []int) []ConstVector {
  n   := len(k)+2
  m   := len(obj.Kmers)+2
  b   := make([]bool, m)
  b[0] = true
  for _, j := range k {
    b[j] = true
  }
  r := make([]ConstVector, len(data))
  for i, _ := range data {
    values  := []float64{1.0}
    indices := []int    {0  }
    for i, j := range k {
      if v := data[i].ValueAt(j); b[j] && v != 0.0 {
        indices = append(indices, i+1)
        values  = append(values , v)
      }
    }
    { // append class label
      indices = append(indices, n-1)
      values  = append(values , data[i].ValueAt(m-1))
    }
    r[i] = NewSparseBareRealVector(indices, values, n)
  }
  return r
}

func (obj *KmerLrOmpEstimator) rankFeatures(data []ConstVector, gamma []float64) []int {
  r := logisticRegression{obj.theta_}
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

  return k
}

func (obj *KmerLrOmpEstimator) selectFeatures(data []ConstVector, gamma []float64) ([]int, bool) {
  m := make(map[int]float64)
  b := false
  z := 0
  // keep all features j with theta_j != 0
  for _, j := range obj.active {
    m[j] = obj.theta_[j]
    if obj.theta_[j] != 0.0 {
      z++
    }
  }
  if z < obj.n {
    for _, j := range obj.rankFeatures(data, gamma) {
      if _, ok := m[j]; !ok {
        // found feature that was not considered
        // in the previous iteration
        b    = true
        m[j] = 1.0
        z   += 1
      }
      if z >= obj.n {
        break
      }
    }
  }
  // convert map to slice
  r := make([]int, z)
  j := 0
  for k, v := range m {
    if v != 0.0 {
      r[j] = k; j++
    }
  }
  sort.Ints(r)
  return r, b
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
  x[0] = 1.0
  return x
}
