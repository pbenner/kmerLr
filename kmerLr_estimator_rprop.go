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

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLrRpropEstimator struct {
  logisticRegression
  Balance       bool
  Epsilon       float64
  MaxIterations int
  Hook          rprop.Hook
  data        []ConstVector
  // list of all features
  Kmers KmerClassList
}

/* -------------------------------------------------------------------------- */

func NewRpropHook(config Config, trace *Trace, icv int, data []ConstVector, estimator *vectorEstimator.LogisticRegression) rprop.Hook {
  hook := NewHook(config, trace, icv, data, estimator)
  k := 0
  f := func(gradient []float64, step []float64, x ConstVector, y Scalar) bool {
    k += 1
    c := 0.0
    for i := 0; i < len(gradient); i++ {
      c += math.Abs(gradient[i])
    }
    hook(x, ConstReal(c), k)
    return false
  }
  return rprop.Hook{f}
}

func NewKmerLrRpropEstimator(config Config, kmers KmerClassList) *KmerLrRpropEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(kmers.Len()+1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    estimator.Balance        = config.Balance
    estimator.L1Reg          = config.Lambda
    estimator.Epsilon        = config.Epsilon
    estimator.StepSizeFactor = config.StepSizeFactor
    if config.MaxEpochs != 0 {
      estimator.MaxIterations = config.MaxEpochs
    }
    r := KmerLrRpropEstimator{}
    r.Balance       = config.Balance
    r.Lambda        = config.Lambda
    r.Epsilon       = config.Epsilon
    r.MaxIterations = config.MaxEpochs
    r.Kmers         = kmers
    r.Theta         = make([]float64, kmers.Len()+1)
    return &r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) Clone() *KmerLrRpropEstimator {
  panic("internal error")
}

func (obj *KmerLrRpropEstimator) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) Estimate(config Config, data []ConstVector) *KmerLr {
  if obj.Balance {
    obj.computeClassWeights(data)
  }
  obj.data = data
  rprop.RunGradient(rprop.DenseGradientF(obj.objectiveGradient), DenseConstRealVector(obj.Theta), 1e-4, []float64{0.5,1.1}, rprop.Epsilon{obj.Epsilon}, rprop.MaxIterations{obj.MaxIterations})
  obj.data = nil
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) objectiveGradient(theta, gradient DenseConstRealVector) error {
  obj.Theta = theta
  obj.Gradient(gradient, obj.data, nil)
  return nil
}

func (obj *KmerLrRpropEstimator) computeClassWeights(data []ConstVector) {
  obj.ClassWeights = compute_class_weights(data)
}
