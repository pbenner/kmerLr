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

/* -------------------------------------------------------------------------- */

type KmerLrEstimatorEnsemble struct {
  *KmerLrEstimator
  Size int
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEnsembleEstimator(config Config, classifier *KmerLr, trace *Trace, icv int) KmerLrEstimatorEnsemble {
  return KmerLrEstimatorEnsemble{NewKmerLrEstimator(config, classifier, trace, icv), config.Ensemble}
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrEstimatorEnsemble) Clone() *KmerLrEstimatorEnsemble {
  panic("internal error")
}

func (obj KmerLrEstimatorEnsemble) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrEstimatorEnsemble) estimate_ensemble(config Config, data_train KmerDataSet) []*KmerLrEnsemble {
  groups := getCvGroups(len(data_train.Data), obj.Size, config.Seed)
  result := make([]*KmerLrEnsemble, len(config.LambdaAuto))
  for i := 0; i < len(result); i++ {
    result[i] = NewKmerLrEnsemble(nil, obj.KmerLrEstimator.KmerLrFeatures)
  }
  for k := 0; k < obj.Size; k++ {
    _, data_train_k := filterCvGroup(data_train, groups, k)
    classifiers, _  := obj.KmerLrEstimator.Estimate(config, data_train_k, KmerDataSet{})
    for i, classifier := range classifiers {
      result[i].AddKmerLr(classifier)
    }
  }
  return result
}

func (obj KmerLrEstimatorEnsemble) Estimate(config Config, data_train, data_test KmerDataSet) ([]*KmerLrEnsemble, [][]float64) {
  if obj.Cooccurrence && config.Copreselection != 0 {
    // reduce data_train and data_test to pre-selected features
    obj.estimate_loop(config, data_train, data_test, config.Copreselection, false)
    data_train.Data  = obj.reduced_data_train.Data
    data_test .Data  = obj.reduced_data_test .Data
    data_train.Kmers = obj.Kmers
    data_test .Kmers = obj.Kmers
  }
  classifiers := obj.estimate_ensemble(config, data_train)
  predictions := make([][]float64, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambda)
    data_test     := classifiers[i].SelectData(config, data_test)
    predictions[i] = classifiers[i].Predict   (config, data_test)
  }
  return classifiers, predictions
}
