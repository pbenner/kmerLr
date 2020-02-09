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
  Cooccurrence bool
  Features     FeatureIndices
  EpsilonLoss  float64
  // reduced data sets
  reduced_data_train []ConstVector
  reduced_data_test  []ConstVector
  labels             []bool
}

/* -------------------------------------------------------------------------- */

func NewScoresLrEstimator(config Config, trace *Trace, icv int, features FeatureIndices) *ScoresLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := ScoresLrEstimator{}
    r.Cooccurrence       = config.Cooccurrence
    r.Features           = features
    r.EpsilonLoss        = config.EpsilonLoss
    r.LogisticRegression = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewScoresHook(config, trace, icv, &r)
    if config.MaxIterations != 0 {
      r.LogisticRegression.MaxIterations = config.MaxIterations
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

func (obj *ScoresLrEstimator) estimate(config Config, data_train []ConstVector, labels []bool) *ScoresLr {
  if err := obj.LogisticRegression.SetSparseData(data_train, labels, len(data_train)); err != nil {
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
    r.LogisticRegression             = *r_.(*vectorDistribution.LogisticRegression)
    r.ScoresLrFeatures.Cooccurrence    = obj   .Cooccurrence
    r.ScoresLrFeatures.Features        = obj   .Features
    return r
  }
}

func (obj *ScoresLrEstimator) estimate_loop(config Config, data_train, data_test []ConstVector, labels []bool, lambda int) *ScoresLr {
  if len(data_train) == 0 {
    return nil
  }
  transform := TransformFull{}
  // estimate transform on full data set so that all estimated
  // classifiers share the same transform
  if !config.NoNormalization {
    transform.Fit(config, append(data_train, data_test...), config.Cooccurrence)
  }
  m, n := obj.n_params(config, data_train, lambda, obj.Cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data_train = make([]ConstVector, len(data_train))
  obj.reduced_data_test  = make([]ConstVector, len(data_test ))
  obj.labels             = labels
  s := newFeatureSelector(config, KmerClassList{}, obj.Cooccurrence, labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*ScoresLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs ; epoch++ {
    // select features on the initial data set
    if r == nil {
      PrintStderr(config, 1, "Selecting %d features... ", n)
    } else {
      d := r.Nonzero()
      PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, selecting %d new features... ", d, n-d)
    }
    selection, lambda, ok := s.Select(data_train, obj.Theta.GetValues(), obj.Features, KmerClassList{}, obj.L1Reg)
    PrintStderr(config, 1, "done\n")
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Theta    = selection.Theta()
    // create actual training and validation data sets
    selection.Data(config, obj.reduced_data_train, data_train)
    selection.Data(config, obj.reduced_data_test , data_test)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e...\n", lambda)
    r = obj.estimate(config, obj.reduced_data_train, labels)
    r.Transform = selection.Transform()
  }
  return r
}

func (obj *ScoresLrEstimator) Estimate(config Config, data_train, data_test []ConstVector, labels []bool) ([]*ScoresLr, [][]float64) {
  classifiers := make([]*ScoresLr, len(config.LambdaAuto))
  predictions := make([][]float64, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambda)
    classifiers[i] = obj.estimate_loop(config, data_train, data_test, labels, lambda)
    predictions[i] = classifiers[i].Predict(config, obj.reduced_data_test)
  }
  return classifiers, predictions
}
