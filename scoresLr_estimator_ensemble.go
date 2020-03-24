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

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type ScoresLrEstimatorEnsemble struct {
  Estimators []*ScoresLrEstimator
  Summary       string
}

/* -------------------------------------------------------------------------- */

func NewScoresLrEnsembleEstimator(config Config, classifier *ScoresLrEnsemble, icv int) ScoresLrEstimatorEnsemble {
  estimators := make([]*ScoresLrEstimator, config.EnsembleSize)
  for i, _ := range estimators {
    estimators[i] = NewScoresLrEstimator(config, classifier.GetComponent(0).Clone(), icv)
  }
  return ScoresLrEstimatorEnsemble{estimators, classifier.Summary}
}

/* -------------------------------------------------------------------------- */

func (obj ScoresLrEstimatorEnsemble) Clone() *ScoresLrEstimatorEnsemble {
  panic("internal error")
}

func (obj ScoresLrEstimatorEnsemble) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj ScoresLrEstimatorEnsemble) GetTrace() Trace {
  trace := Trace{}
  for _, estimator := range obj.Estimators {
    trace.AppendTrace(estimator.trace)
  }
  return trace
}

/* -------------------------------------------------------------------------- */

func (obj ScoresLrEstimatorEnsemble) estimate_ensemble(config Config, data_train ScoresDataSet, transform TransformFull) []*ScoresLrEnsemble {
  groups := getCvGroups(len(data_train.Data), config.EnsembleSize, config.Seed)
  result := make([]*ScoresLrEnsemble, len(config.LambdaAuto))
  for i := 0; i < len(result); i++ {
    result[i] = NewScoresLrEnsemble(obj.Summary)
  }
  classifiers := make([][]*ScoresLr, config.EnsembleSize)
  config.Pool.RangeJob(0, config.EnsembleSize, func(k int, pool threadpool.ThreadPool, erf func() error) error {
    config := config; config.Pool = pool

    _, data_train_k := scoresFilterCvGroup(data_train, groups, k)
    classifiers[k] = obj.Estimators[k].Estimate(config, data_train_k, transform)
    return nil
  })
  for k := 0; k < len(classifiers); k++ {
    for i, classifier := range classifiers[k] {
      if err := result[i].AddScoresLr(classifier); err != nil {
        panic("internal error")
      }
    }
  }
  return result
}

func (obj ScoresLrEstimatorEnsemble) Estimate(config Config, data_train, data_test ScoresDataSet) ([]*ScoresLrEnsemble, [][]float64) {
  if obj.Estimators[0].Cooccurrence && config.Copreselection != 0 {
    transform := TransformFull{}
    // estimate transform on full data set so that all estimated
    // classifiers share the same transform
    if !config.NoNormalization {
      transform.Fit(config, append(data_train.Data, data_test.Data...), false)
    }
    // reduce data_train and data_test to pre-selected features
    r, r_data := obj.Estimators[0].estimate_loop(config, data_train, transform, config.Copreselection, false)
    data_train.Data  = r_data
    data_train.Index = r.Index
    // reset all estimators
    for _, estimator := range obj.Estimators {
      estimator.Reset()
    }
  }
  transform := TransformFull{}
  if !config.NoNormalization {
    transform.Fit(config, append(data_train.Data, data_test.Data...), obj.Estimators[0].Cooccurrence)
  }
  classifiers := obj.estimate_ensemble(config, data_train, transform)
  predictions := make([][]float64, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambda)
    data_test     := classifiers[i].SelectData(config, data_test)
    predictions[i] = classifiers[i].Predict   (config, data_test)
  }
  return classifiers, predictions
}
