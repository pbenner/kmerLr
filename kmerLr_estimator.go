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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLrEstimator struct {
  vectorEstimator.LogisticRegression
  KmerLrFeatures
  EpsilonLoss  float64
  reduced_data KmerDataSet
  trace        Trace
  path         KmerRegularizationPath
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, classifier *KmerLr, icv int) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLrEstimator{}
    r.KmerLrFeatures                    = classifier.KmerLrFeatures
    r.EpsilonLoss                       = config.EpsilonLoss
    r.LogisticRegression                = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, icv, r)
    if config.MaxIterations != 0 {
      r.LogisticRegression.MaxIterations = config.MaxIterations
    }
    if len(classifier.Theta) > 0 {
      r.LogisticRegression.Theta = DenseFloat64Vector(classifier.Theta)
    }
    return r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) Clone() *KmerLrEstimator {
  panic("internal error")
}

func (obj *KmerLrEstimator) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) Reset() {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
  } else {
    obj.LogisticRegression = *estimator
    obj.Features           = FeatureIndices{}
    obj.Kmers              = KmerClassList {}
  }
}

func (obj *KmerLrEstimator) n_params(config Config, data []ConstVector, lambdaAuto int, cooccurrence bool) (int, int) {
  if m := data[0].Dim()-1; lambdaAuto != 0 {
    return m, lambdaAuto
  } else {
    if cooccurrence {
      return m, CoeffIndex(m).Dim()
    } else {
      return m, m
    }
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) estimate_debug_gradient(config Config, data KmerDataSet) {
  r := NewReal64(0.0)
  s := NewReal64(0.0)
  t := AsDenseReal64Vector(obj.Theta)
  t.Variables(1)
  lr, _ := vectorDistribution.NewLogisticRegression(t)
  for i := 0; i < len(data.Data); i++ {
    lr.ClassLogPdf(s, data.Data[i], data.Labels[i])
    if data.Labels[i] {
      s.Mul(s, ConstFloat64(obj.ClassWeights[1]))
    } else {
      s.Mul(s, ConstFloat64(obj.ClassWeights[0]))
    }
    r.Add(r, s)
  }
  r.Neg(r)
  r.Div(r, ConstFloat64(float64(len(data.Data))))
  fmt.Printf("gradient (autodiff): %v\n\n", GetGradient(Float64Type, r))
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) estimate(config Config, data KmerDataSet, transform Transform, cooccurrence bool, debug bool) *KmerLr {
  transform.Apply(config, data.Data)
  if debug {
    obj.estimate_debug_gradient(config, data)
  }
  if err := obj.LogisticRegression.SetSparseData(data.Data, data.Labels, len(data.Data)); err != nil {
    log.Fatal(err)
  }
  if err := obj.LogisticRegression.Estimate(nil, config.PoolSaga); err != nil {
    log.Fatal(err)
  }
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{}
    r.Theta          = r_.(*vectorDistribution.LogisticRegression).Theta.(DenseFloat64Vector)
    r.KmerLrFeatures = obj.KmerLrFeatures
    r.Cooccurrence   = cooccurrence
    r.Transform      = transform
    if config.SavePath {
      obj.path.Append(-1, obj.L1Reg/float64(len(data.Data)), r.KmerLrFeatures.Kmers, r.Theta[1:])
    }
    return r
  }
}

func (obj *KmerLrEstimator) estimate_fixed(config Config, data KmerDataSet, transform TransformFull, cooccurrence bool) *KmerLr {
  if len(data.Data) == 0 {
    return nil
  }
  if len(data.Kmers) != data.Data[0].Dim()-1 {
    panic("internal error")
  }
  m, _ := obj.n_params(config, data.Data, 0, cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data.Data   = make([]ConstVector, len(data.Data))
  obj.reduced_data.Labels = data.Labels
  s := newFeatureSelector(config, data.Kmers, nil, nil, cooccurrence, data.Labels, transform, obj.ClassWeights, m, 0, config.EpsilonLambda)
  r := (*KmerLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs; epoch++ {
    selection, ok := s.SelectFixed(data.Data, AsDenseFloat64Vector(obj.Theta), obj.Features, obj.Kmers, nil, nil, config.Lambda, config.MaxFeatures)
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = config.Lambda*float64(len(data.Data))
    obj.Features = selection.Features()
    obj.Kmers    = selection.Kmers()
    obj.Theta    = selection.Theta()
    // create actual training data set
    selection.Data(config, obj.reduced_data.Data, data.Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e and %d features...\n", config.Lambda, len(obj.Features))
    r = obj.estimate(config, obj.reduced_data, selection.Transform(), cooccurrence, false)
  }
  obj.reduced_data = KmerDataSet{}
  return r
}

func (obj *KmerLrEstimator) estimate_loop(config Config, data KmerDataSet, transform TransformFull, lambdaAuto int, cooccurrence bool) *KmerLr {
  if len(data.Data) == 0 {
    return nil
  }
  if len(data.Kmers) != data.Data[0].Dim()-1 {
    panic("internal error")
  }
  debug := false
  m, n := obj.n_params(config, data.Data, lambdaAuto, cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data.Data   = make([]ConstVector, len(data.Data))
  obj.reduced_data.Labels = data.Labels
  s := newFeatureSelector(config, data.Kmers, nil, nil, cooccurrence, data.Labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*KmerLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs; epoch++ {
    // select features on the initial data set
    if r != nil {
      d := r.Nonzero()
      if d > n {
        PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, removing %d features...\n", d, d-n)
      } else {
        PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, selecting %d new features...\n", d, n-d)
      }
    }
    selection, lambda, ok := s.Select(data.Data, obj.Theta, obj.Features, obj.Kmers, nil, nil, obj.L1Reg, debug)
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda*float64(len(data.Data))
    obj.Features = selection.Features()
    obj.Kmers    = selection.Kmers()
    obj.Theta    = selection.Theta()
    // create actual training data set
    selection.Data(config, obj.reduced_data.Data, data.Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e...\n", lambda)
    r = obj.estimate(config, obj.reduced_data, selection.Transform(), cooccurrence, debug)
  }
  obj.reduced_data = KmerDataSet{}
  return r
}

func (obj *KmerLrEstimator) Estimate(config Config, data KmerDataSet, transform TransformFull) []*KmerLr {
  if !math.IsNaN(config.Lambda) {
    classifiers   := make([]*KmerLr, 1)
    classifiers[0] = obj.estimate_fixed(config, data, transform, obj.Cooccurrence)
    return classifiers
  } else {
    classifiers := make([]*KmerLr, len(config.LambdaAuto))
    for i, lambdaAuto := range config.LambdaAuto {
      PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambdaAuto)
      classifiers[i] = obj.estimate_loop(config, data, transform, lambdaAuto, obj.Cooccurrence)
    }
    return classifiers
  }
}
