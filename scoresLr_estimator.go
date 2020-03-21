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
      r.SetParameters(NewDenseBareRealVector(classifier.Theta))
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

func (obj *ScoresLrEstimator) estimate(config Config, data ScoresDataSet, transform Transform, cooccurrence bool) *ScoresLr {
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
    r.Theta            = r_.(*vectorDistribution.LogisticRegression).Theta.GetValues()
    r.ScoresLrFeatures = obj.ScoresLrFeatures
    r.Cooccurrence     = cooccurrence
    r.Transform        = transform
    return r
  }
}

func (obj *ScoresLrEstimator) estimate_loop(config Config, data ScoresDataSet, transform TransformFull, lambda int, cooccurrence bool) (*ScoresLr, []ConstVector) {
  if len(data.Data) == 0 {
    return nil, nil
  }
  m, n := obj.n_params(config, data.Data, lambda, obj.Cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data.Data   = make([]ConstVector, len(data.Data))
  obj.reduced_data.Labels = data.Labels
  s := newFeatureSelector(config, KmerClassList{}, data.Index, cooccurrence, data.Labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*ScoresLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs ; epoch++ {
    // select features on the initial data set
    if r == nil {
      PrintStderr(config, 1, "Selecting %d features... ", n)
    } else {
      d := r.Nonzero()
      if d > n {
        PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, removing %d features... ", d, d-n)
      } else {
        PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, selecting %d new features... ", d, n-d)
      }
    }
    selection, lambda, ok := s.Select(data.Data, obj.Theta.GetValues(), obj.Features, KmerClassList{}, obj.Index, obj.L1Reg)
    PrintStderr(config, 1, "done\n")
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Index    = selection.Index()
    obj.Theta    = selection.Theta()
    // create actual training data sets
    selection.Data(config, obj.reduced_data.Data, data.Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e...\n", lambda)
    r = obj.estimate(config, obj.reduced_data, selection.Transform(), cooccurrence)
  }
  r_data := obj.reduced_data.Data
  obj.reduced_data = ScoresDataSet{}
  return r, r_data
}

func (obj *ScoresLrEstimator) Estimate(config Config, data ScoresDataSet, transform TransformFull) []*ScoresLr {
  classifiers := make([]*ScoresLr, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambda)
    classifiers[i], _ = obj.estimate_loop(config, data, transform, lambda, obj.Cooccurrence)
  }
  return classifiers
}
