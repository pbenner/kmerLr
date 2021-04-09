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
import   "math"

import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type KmerLrEstimatorEnsemble struct {
  Estimators []*KmerLrEstimator
  Summary       string
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEnsembleEstimator(config Config, classifier *KmerLrEnsemble, icv int) KmerLrEstimatorEnsemble {
  estimators := make([]*KmerLrEstimator, config.EnsembleSize)
  for i, _ := range estimators {
    estimators[i] = NewKmerLrEstimator(config, classifier.GetComponent(0).Clone(), icv)
  }
  return KmerLrEstimatorEnsemble{estimators, classifier.Summary}
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrEstimatorEnsemble) Clone() *KmerLrEstimatorEnsemble {
  panic("internal error")
}

func (obj KmerLrEstimatorEnsemble) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrEstimatorEnsemble) GetTrace() Trace {
  trace := Trace{}
  for _, estimator := range obj.Estimators {
    trace.AppendTrace(estimator.trace)
  }
  return trace
}

func (obj KmerLrEstimatorEnsemble) GetPath() KmerRegularizationPath {
  path := KmerRegularizationPath{}
  for i, estimator := range obj.Estimators {
    path.AppendPath(i, estimator.path)
  }
  return path
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrEstimatorEnsemble) estimate_ensemble(config Config, data_train KmerDataSet, transform TransformFull) []*KmerLrEnsemble {
  groups, _ := getCvGroups(len(data_train.Data), config.EnsembleSize, config.ValidationSize, config.Seed)
  result    := make([]*KmerLrEnsemble, len(config.LambdaAuto))
  for i := 0; i < len(result); i++ {
    result[i] = NewKmerLrEnsemble(obj.Summary)
  }
  classifiers := make([][]*KmerLr, config.EnsembleSize)
  config.Pool.RangeJob(0, config.EnsembleSize, func(k int, pool threadpool.ThreadPool, erf func() error) error {
    config := config; config.Pool = pool

    _, _, data_k := filterCvGroup(data_train, groups, nil, k)
    classifiers[k] = obj.Estimators[k].Estimate(config, data_k, transform)
    return nil
  })
  for k := 0; k < len(classifiers); k++ {
    for i, classifier := range classifiers[k] {
      if err := result[i].AddKmerLr(classifier); err != nil {
        panic("internal error")
      }
    }
  }
  return result
}

func (obj KmerLrEstimatorEnsemble) Estimate(config Config, data_train, data_val, data_test KmerDataSet) ([]*KmerLrEnsemble, [][]float64) {
  if obj.Estimators[0].Cooccurrence && config.Copreselection != 0 {
    transform := TransformFull{}
    // estimate transform on full data set so that all estimated
    // classifiers share the same transform
    transform.Fit(config, append(data_train.Data, data_test.Data...), false)
    // reduce data_train and data_test to pre-selected features
    r := obj.Estimators[0].estimate_loop(config, data_train, transform, config.Copreselection, false)
    r.Transform      = Transform{}
    data_train.Data  = r.SelectData(config, data_train)
    data_train.Kmers = r.Kmers
    data_test .Data  = r.SelectData(config, data_test)
    data_test .Kmers = r.Kmers
    // reset all estimators
    for _, estimator := range obj.Estimators {
      estimator.Reset()
    }
  }
  transform := TransformFull{}
  transform.Fit(config, append(data_train.Data, data_test.Data...), obj.Estimators[0].Cooccurrence)
  classifiers := obj.estimate_ensemble(config, data_train, transform)
  // if validation data is available, select best classifier...
  if len(data_val.Data) > 0 {
    i_best := 0
    v_best := math.Inf(1)
    for i, _ := range classifiers {
      d := classifiers[i].SelectData(config, data_val)
      v := classifiers[i].LossAvrg(config, d, data_val.Labels)
      if v < v_best {
        i_best = i
        v_best = v
      }
      PrintStderr(config, 1, "> Classifier %d has loss %v...\n", i, v)
    }
    PrintStderr(config, 1, "> Selecting classifier %d\n", i_best)
    classifiers = []*KmerLrEnsemble{classifiers[i_best]}
  }
  // evaluate classifier(s) on test data
  predictions := make([][]float64, len(classifiers))
  for i, _ := range classifiers {
    data_test_    := classifiers[i].SelectData(config, data_test)
    predictions[i] = classifiers[i].Predict   (config, data_test_)
  }
  return classifiers, predictions
}
