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
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type ScoresLrEstimator struct {
  vectorEstimator.LogisticRegression
  ScoresLrFeatures
  EpsilonLoss  float64
  reduced_data ScoresDataSet
  trace        Trace
}

/* -------------------------------------------------------------------------- */

func NewScoresLrEstimator(config Config, classifier *ScoresLr, icv int) *ScoresLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := ScoresLrEstimator{}
    r.ScoresLrFeatures                  = classifier.ScoresLrFeatures
    r.EpsilonLoss                       = config.EpsilonLoss
    r.LogisticRegression                = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewScoresHook(config, icv, &r)
    if config.MaxIterations != 0 {
      r.LogisticRegression.MaxIterations = config.MaxIterations
    }
    if len(classifier.Theta) > 0 {
      r.SetParameters(NewDenseFloat64Vector(classifier.Theta))
    }
    return &r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEstimator) Clone() *ScoresLrEstimator {
  panic("internal error")
}

func (obj *ScoresLrEstimator) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEstimator) Reset() {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
  } else {
    obj.LogisticRegression = *estimator
    obj.Features           = FeatureIndices{}
    obj.Index              = []int{}
    obj.Names              = []string{}
  }
}

func (obj *ScoresLrEstimator) n_params(config Config, data []ConstVector, lambdaAuto int, cooccurrence bool) (int, int) {
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

func (obj *ScoresLrEstimator) estimate(config Config, data ScoresDataSet, transform Transform, cooccurrence bool) *ScoresLr {
  transform.Apply(config, data.Data)
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
    r := &ScoresLr{}
    r.Theta            = r_.(*vectorDistribution.LogisticRegression).Theta.(DenseFloat64Vector)
    r.ScoresLrFeatures = obj.ScoresLrFeatures
    r.Cooccurrence     = cooccurrence
    r.Transform        = transform
    return r
  }
}

func (obj *ScoresLrEstimator) estimate_fixed(config Config, data ScoresDataSet, transform TransformFull, cooccurrence bool) *ScoresLr {
  if len(data.Data) == 0 {
    return nil
  }
  m, _ := obj.n_params(config, data.Data, 0, obj.Cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data.Data   = make([]ConstVector, len(data.Data))
  obj.reduced_data.Labels = data.Labels
  s := newFeatureSelector(config, KmerClassList{}, data.Index, data.Names, cooccurrence, data.Labels, transform, obj.ClassWeights, m, 0, config.EpsilonLambda)
  r := (*ScoresLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs; epoch++ {
    selection, ok := s.SelectFixed(data.Data, obj.Theta, obj.Features, KmerClassList{}, obj.Index, obj.Names, config.Lambda, config.MaxFeatures)
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = config.Lambda
    obj.Features = selection.Features()
    obj.Index    = selection.Index()
    obj.Names    = selection.Names()
    obj.Theta    = selection.Theta()
    // create actual training data sets
    selection.Data(config, obj.reduced_data.Data, data.Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e and %d features...\n", config.Lambda, len(obj.Features))
    r = obj.estimate(config, obj.reduced_data, selection.Transform(), cooccurrence)
  }
  obj.reduced_data = ScoresDataSet{}
  return r
}

func (obj *ScoresLrEstimator) estimate_loop(config Config, data ScoresDataSet, transform TransformFull, lambdaAuto int, cooccurrence bool) *ScoresLr {
  if len(data.Data) == 0 {
    return nil
  }
  m, n := obj.n_params(config, data.Data, lambdaAuto, obj.Cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data.Data   = make([]ConstVector, len(data.Data))
  obj.reduced_data.Labels = data.Labels
  s := newFeatureSelector(config, KmerClassList{}, data.Index, data.Names, cooccurrence, data.Labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*ScoresLr)(nil)
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
    selection, lambda, ok := s.Select(data.Data, obj.Theta, obj.Features, KmerClassList{}, obj.Index, obj.Names, obj.L1Reg, false)
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Index    = selection.Index()
    obj.Names    = selection.Names()
    obj.Theta    = selection.Theta()
    // create actual training data sets
    selection.Data(config, obj.reduced_data.Data, data.Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e...\n", lambda)
    r = obj.estimate(config, obj.reduced_data, selection.Transform(), cooccurrence)
  }
  obj.reduced_data = ScoresDataSet{}
  return r
}

func (obj *ScoresLrEstimator) Estimate(config Config, data ScoresDataSet, transform TransformFull) []*ScoresLr {
  if config.Lambda != 0.0 {
    classifiers   := make([]*ScoresLr, 1)
    classifiers[0] = obj.estimate_fixed(config, data, transform, obj.Cooccurrence)
    return classifiers
  } else {
    classifiers := make([]*ScoresLr, len(config.LambdaAuto))
    for i, lambdaAuto := range config.LambdaAuto {
      PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambdaAuto)
      classifiers[i] = obj.estimate_loop(config, data, transform, lambdaAuto, obj.Cooccurrence)
    }
    return classifiers
  }
}
