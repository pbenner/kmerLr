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
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, kmers KmerClassList, trace *Trace, icv int, data []ConstVector, features FeatureIndices, labels []bool) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLrEstimator{}
    r.Cooccurrence       = config.Cooccurrence
    r.Kmers              = kmers
    r.Features           = features
    r.EpsilonLoss        = config.EpsilonLoss
    r.LogisticRegression = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, trace, icv, data, labels, &r)
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
  if len(obj.Kmers) != data[0].Dim()-1 {
    panic("internal error")
  }
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

func (obj *KmerLrEstimator) estimate_debug(config Config, data_train []ConstVector, labels []bool) *KmerLr {
  theta := obj.Theta.GetValues()
  gamma := 0.0001
  lr1   := logisticRegression{theta, obj.ClassWeights, 0.0      , false, TransformFull{}, config.Pool}
  lr2   := logisticRegression{theta, obj.ClassWeights, obj.L1Reg, false, TransformFull{}, config.Pool}
  for i := 0; i < 10000; i++ {
    g := lr1.Gradient(nil, data_train, labels)
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
    PrintStderr(config, 2, "loss: %f\n", lr2.Loss(data_train, labels))
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

func (obj *KmerLrEstimator) estimate(config Config, data_train []ConstVector, labels []bool) *KmerLr {
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

func (obj *KmerLrEstimator) estimate_loop(config Config, data_train, data_test []ConstVector, labels []bool, transform TransformFull, lambdaAuto int, cooccurrence bool) *KmerLr {
  if len(data_train) == 0 {
    return nil
  }
  m, n := obj.n_params(config, data_train, lambdaAuto, cooccurrence)
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
  s := newFeatureSelector(config, obj.Kmers, cooccurrence, labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*KmerLr)(nil)
  for epoch := 0; config.MaxEpochs == 0 || epoch < config.MaxEpochs ; epoch++ {
    // select features on the initial data set
    if r == nil {
      PrintStderr(config, 1, "Selecting %d features... ", n)
    } else {
      d := r.Nonzero()
      PrintStderr(config, 1, "Estimated classifier has %d non-zero coefficients, selecting %d new features... ", d, n-d)
    }
    selection, lambda, ok := s.Select(copy_data_train, obj.Theta.GetValues(), obj.Features, obj.Kmers, obj.L1Reg)
    PrintStderr(config, 1, "done\n")
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Kmers    = selection.Kmers()
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

func (obj *KmerLrEstimator) Estimate(config Config, data_train, data_test []ConstVector, labels []bool, transform TransformFull) []*KmerLr {
  if obj.Cooccurrence && config.Copreselection != 0 {
    // reduce data_train and data_test to pre-selected features
    obj.estimate_loop(config, data_train, data_test, labels, transform, config.Copreselection, false)
  }
  r := make([]*KmerLr, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    r[i] = obj.estimate_loop(config, data_train, data_test, labels, transform, lambda, obj.Cooccurrence)
  }
  return r
}
