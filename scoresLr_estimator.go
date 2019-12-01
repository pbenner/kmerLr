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
}

/* -------------------------------------------------------------------------- */

func NewScoresLrEstimator(config Config, trace *Trace, icv int, data []ConstVector, features FeatureIndices, labels []bool) *ScoresLrEstimator {
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
    r.LogisticRegression.Hook           = NewScoresHook(config, trace, icv, data, labels, &r)
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

func (obj *ScoresLrEstimator) n_params(config Config, data []ConstVector) (int, int) {
  if m := data[0].Dim()-1; config.LambdaAuto != 0 {
    return m, config.LambdaAuto
  } else {
    if config.Cooccurrence {
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

func (obj *ScoresLrEstimator) Estimate(config Config, data_train, data_test []ConstVector, labels []bool, transform TransformFull) *ScoresLr {
  if len(data_train) == 0 {
    return nil
  }
  m, n := obj.n_params(config, data_train)
  // compute class weights
  obj.LogisticRegression.SetLabels(labels)
  // create a copy of data arrays, from which to select subsets
  copy_data_train := make([]ConstVector, len(data_train))
  copy_data_test  := make([]ConstVector, len(data_test))
  for i, x := range data_train {
    copy_data_train[i] = x
  }
  for i, x := range data_test {
    copy_data_test [i] = x
  }
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
    selection, lambda, ok := s.Select(copy_data_train, obj.Theta.GetValues(), obj.Features, KmerClassList{}, obj.L1Reg)
    PrintStderr(config, 1, "done\n")
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Theta    = selection.Theta()
    // create actual training and validation data sets
    selection.Data(config, data_train, copy_data_train)
    selection.Data(config, data_test , copy_data_test)

    PrintStderr(config, 1, "Estimating parameters with lambda=%f...\n", lambda)
    r = obj.estimate(config, data_train, labels)
    r.Transform = selection.Transform()
  }
  return r
}