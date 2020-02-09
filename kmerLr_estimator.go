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
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLrEstimator struct {
  vectorEstimator.LogisticRegression
  Cooccurrence bool
  Kmers        KmerClassList
  Features     FeatureIndices
  EpsilonLoss  float64
  // reduced data sets
  reduced_data_train KmerDataSet
  reduced_data_test  KmerDataSet
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, trace *Trace, icv int) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLrEstimator{}
    r.Cooccurrence       = config.Cooccurrence
    r.EpsilonLoss        = config.EpsilonLoss
    r.LogisticRegression = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, trace, icv, &r)
    if config.MaxIterations != 0 {
      r.LogisticRegression.MaxIterations = config.MaxIterations
    }
    return &r
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

func (obj *KmerLrEstimator) estimate_debug(config Config, data_train KmerDataSet) *KmerLr {
  theta := obj.Theta.GetValues()
  gamma := 0.0001
  lr1   := logisticRegression{theta, obj.ClassWeights, 0.0      , false, TransformFull{}, config.Pool}
  lr2   := logisticRegression{theta, obj.ClassWeights, obj.L1Reg, false, TransformFull{}, config.Pool}
  for i := 0; i < 10000; i++ {
    g := lr1.Gradient(nil, data_train.Data, data_train.Labels)
    for k := 0; k < len(theta); k++ {
      theta[k] = theta[k] - gamma*g[k]
      if k > 0 {
        if theta[k] >= 0.0 {
          theta[k] =  math.Max(math.Abs(theta[k]) - gamma*obj.L1Reg, 0.0)
        } else {
          theta[k] = -math.Max(math.Abs(theta[k]) - gamma*obj.L1Reg, 0.0)
        }
      }
    }
    PrintStderr(config, 2, "loss: %f\n", lr2.Loss(data_train.Data, data_train.Labels))
  }
  obj.Theta = NewDenseBareRealVector(theta)
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{}
    r.LogisticRegression             = *r_.(*vectorDistribution.LogisticRegression)
    r.KmerLrFeatures.Binarize        = config.Binarize
    r.KmerLrFeatures.Cooccurrence    = obj   .Cooccurrence
    r.KmerLrFeatures.Features        = obj   .Features
    r.KmerLrFeatures.Kmers           = obj   .Kmers
    r.KmerLrFeatures.KmerEquivalence = config.KmerEquivalence
    return r
  }
}

func (obj *KmerLrEstimator) estimate(config Config, data_train KmerDataSet) *KmerLr {
  if err := obj.LogisticRegression.SetSparseData(data_train.Data, data_train.Labels, len(data_train.Data)); err != nil {
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
    r.LogisticRegression             = *r_.(*vectorDistribution.LogisticRegression)
    r.KmerLrFeatures.Binarize        = config.Binarize
    r.KmerLrFeatures.Cooccurrence    = obj   .Cooccurrence
    r.KmerLrFeatures.Features        = obj   .Features
    r.KmerLrFeatures.Kmers           = obj   .Kmers
    r.KmerLrFeatures.KmerEquivalence = config.KmerEquivalence
    return r
  }
}

func (obj *KmerLrEstimator) estimate_loop(config Config, data_train, data_test KmerDataSet, lambdaAuto int, cooccurrence bool) *KmerLr {
  if len(data_train.Data) == 0 {
    return nil
  }
  if len(data_train.Kmers) != data_train.Data[0].Dim()-1 {
    panic("internal error")
  }
  transform := TransformFull{}
  // estimate transform on full data set so that all estimated
  // classifiers share the same transform
  if !config.NoNormalization {
    transform.Fit(config, append(data_train.Data, data_test.Data...), cooccurrence)
  }
  m, n := obj.n_params(config, data_train.Data, lambdaAuto, cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data_train.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data_train.Data   = make([]ConstVector, len(data_train.Data))
  obj.reduced_data_test .Data   = make([]ConstVector, len(data_test .Data))
  obj.reduced_data_train.Labels = data_train.Labels
  obj.reduced_data_test .Labels = data_test .Labels
  s := newFeatureSelector(config, data_train.Kmers, cooccurrence, data_train.Labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*KmerLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs ; epoch++ {
    // select features on the initial data set
    if r == nil {
      PrintStderr(config, 1, "Selecting %d features... ", n)
    } else {
      d := r.Nonzero()
      PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, selecting %d new features... ", d, n-d)
    }
    selection, lambda, ok := s.Select(data_train.Data, obj.Theta.GetValues(), obj.Features, obj.Kmers, obj.L1Reg)
    PrintStderr(config, 1, "done\n")
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Kmers    = selection.Kmers()
    obj.Theta    = selection.Theta()
    // create actual training and validation data sets
    selection.Data(config, obj.reduced_data_train.Data, data_train.Data)
    selection.Data(config, obj.reduced_data_test .Data, data_test .Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e...\n", lambda)
    r = obj.estimate(config, obj.reduced_data_train)
    r.Transform = selection.Transform()
  }
  return r
}

func (obj *KmerLrEstimator) Estimate(config Config, data_train, data_test KmerDataSet) ([]*KmerLr, [][]float64) {
  if obj.Cooccurrence && config.Copreselection != 0 {
    // reduce data_train and data_test to pre-selected features
    obj.estimate_loop(config, data_train, data_test, config.Copreselection, false)
    data_train.Data  = obj.reduced_data_train.Data
    data_test .Data  = obj.reduced_data_test .Data
    data_train.Kmers = obj.Kmers
    data_test .Kmers = obj.Kmers
  }
  classifiers := make(  []*KmerLr, len(config.LambdaAuto))
  predictions := make([][]float64, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambda)
    classifiers[i] = obj.estimate_loop(config, data_train, data_test, lambda, obj.Cooccurrence)
    predictions[i] = classifiers[i].Predict(config, obj.reduced_data_test.Data)
  }
  return classifiers, predictions
}
