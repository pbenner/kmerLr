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
import   "sort"

import . "github.com/pbenner/ngstat/estimation"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"

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
  // number of omp iterations
  OmpIterations int
}

/* -------------------------------------------------------------------------- */

func NewKmerLrOmpEstimator(config Config, kmers KmerClassList) *KmerLrOmpEstimator {
  n := kmers.Len() + 1
  if config.Cooccurrence {
    n = (kmers.Len()+1)*kmers.Len()/2 + 1
  }
  if estimator, err := vectorEstimator.NewLogisticRegression(n, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    estimator.Balance        = config.Balance
    estimator.Seed           = config.Seed
    estimator.L1Reg          = config.Lambda
    estimator.Epsilon        = config.Epsilon
    estimator.StepSizeFactor = config.StepSizeFactor
    if config.MaxEpochs != 0 {
      estimator.MaxIterations = config.MaxEpochs
    }
    r := KmerLrOmpEstimator{}
    r.LogisticRegression = *estimator
    r.Kmers         = kmers
    r.theta_        = make([]float64, kmers.Len()+1)
    r.OmpIterations = config.OmpIterations
    r.n             = config.Omp
    return &r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) Clone() *KmerLrOmpEstimator {
  panic("internal error")
}

func (obj *KmerLrOmpEstimator) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) Estimate(config Config, data []ConstVector) *KmerLr {
  if obj.Balance {
    obj.computeClassWeights(data)
  }
  gamma := obj.normalizationConstants(config, data)
  for iOmp := 0; iOmp < obj.OmpIterations; iOmp++ {
    for i := 1; i <= obj.n; i++ {
      if len(obj.active) >= i {
        // remove feature
        obj.theta_[obj.active[i-1]+1] = 0
        obj.active = append(obj.active[0:i-1], obj.active[i:]...)
        obj.printActive()
      }
      // select a subset of features using OMP
      features := obj.selectFeatures(data, gamma, len(obj.active)+1)
      // get subset of data and coefficients for these features
      features_data := obj.selectData(data, features)
      // estimate reduced set of coefficients
      if err := EstimateOnSingleTrackConstData(config.SessionConfig, &obj.LogisticRegression, features_data); err != nil {
        log.Fatal(err)
      }
      // copy coefficients to backup vector
      obj.saveCoefficients(features)
    }
  }
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{LogisticRegression: *r_.(*vectorDistribution.LogisticRegression)}
    r.KmerLrAlphabet.Binarize        = config.Binarize
    r.KmerLrAlphabet.KmerEquivalence = config.KmerEquivalence
    r.KmerLrAlphabet.Kmers           = make(KmerClassList, len(obj.active))
    for l, j := range obj.active {
      r.KmerLrAlphabet.Kmers[l] = obj.Kmers[j]
    }
    return r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrOmpEstimator) printActive() {
  fmt.Println("Active features:")
  for _, k := range obj.active {
    fmt.Printf(": %e %s\n", obj.theta_[k+1], obj.Kmers[k])
  }
}

func (obj *KmerLrOmpEstimator) saveCoefficients(k []int) {
  for j, _ := range obj.theta_ {
    obj.theta_[j] = 0.0
  }
  for i, j := range k {
    obj.theta_[j+1] = obj.Theta.ValueAt(i+1)
  }
  obj.theta_[0] = obj.Theta.ValueAt(0)
}

func (obj *KmerLrOmpEstimator) selectCoefficients(k []int) Vector {
  v := make([]float64, len(k)+1)
  for i, j := range k {
    v[i+1] = obj.theta_[j+1]
  }
  v[0] = obj.theta_[0]
  return NewDenseBareRealVector(v)
}

func (obj *KmerLrOmpEstimator) selectData(data []ConstVector, k []int) []ConstVector {
  n := len(k)+2
  m := len(obj.Kmers)+2
  b := make([]bool, m-2)
  for _, j := range k {
    b[j] = true
  }
  r := make([]ConstVector, len(data))
  for i, _ := range data {
    values  := []ConstReal{1.0}
    indices := []int      {0  }
    for l, j := range k {
      if v := data[i].ValueAt(j+1); b[j] && v != 0.0 {
        indices = append(indices, l+1)
        values  = append(values , ConstReal(v))
      }
    }
    if j, v := data[i].(SparseConstRealVector).Last(); j != m-1 {
      panic("internal error")
    } else {
      // append class label
      indices = append(indices, n-1)
      values  = append(values , v)
    }
    r[i] = UnsafeSparseConstRealVector(indices, values, n)
  }
  return r
}

func (obj *KmerLrOmpEstimator) rankFeatures(data []ConstVector, gamma []float64) []int {
  r := logisticRegression{}
  r.Theta        = obj.theta_
  r.Lambda       = 0.0
  r.ClassWeights = obj.ClassWeights
  g := r.Gradient(nil, data, gamma)
  if len(g) != len(obj.Kmers)+1 {
    panic("internal error")
  }
  g  = g[1:]
  k := make([]int, len(g))
  for i := 0; i < len(g); i++ {
    g[i] = math.Abs(g[i])
    k[i] = i
  }
  FloatInt{g, k}.SortReverse()
  return k
}

func (obj *KmerLrOmpEstimator) setActive(k []int) {
  coefficients := obj.selectCoefficients(k)
  // copy coefficients to estimator
  if err := obj.LogisticRegression.SetParameters(coefficients); err != nil {
    log.Fatal(err)
  }
  obj.active = k
}

func (obj *KmerLrOmpEstimator) selectFeatures(data []ConstVector, gamma []float64, n int) []int {
  m := make(map[int]struct{})
  // keep all features j with theta_j != 0
  for _, j := range obj.active {
    m[j] = struct{}{}
  }
  if len(m) < n {
    for _, j := range obj.rankFeatures(data, gamma) {
      if _, ok := m[j]; !ok {
        m[j] = struct{}{}
      }
      if len(m) >= n {
        break
      }
    }
  }
  // convert map to slice
  r := make([]int, len(m))
  j := 0
  for k, _ := range m {
    r[j] = k; j++
  }
  sort.Ints(r)
  // set coefficients
  obj.setActive(r)
  return r
}

func (obj *KmerLrOmpEstimator) computeClassWeights(data []ConstVector) {
  obj.ClassWeights = compute_class_weights(data)
}

func (obj *KmerLrOmpEstimator) normalizationConstants(config Config, data []ConstVector) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()
  x := make([]float64, m-1)
  PrintStderr(config, 1, "Computing OMP normalization... ")
  for i := 0; i < n; i++ {
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 && j != m-1 {
        x[j] += it.GetConst().GetValue()*it.GetConst().GetValue()
      }
    }
  }
  PrintStderr(config, 1, "done\n")
  x[0] = 1.0
  return x
}
