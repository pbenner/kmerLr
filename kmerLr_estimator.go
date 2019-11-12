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

type KmerLrEstimator struct {
  vectorEstimator.LogisticRegression
  Cooccurrence bool
  Kmers        KmerClassList
  Transform    Transform
  Features     FeatureIndices
  EpsilonLoss  float64
  iterations   int
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, kmers KmerClassList, trace *Trace, icv int, data []ConstVector, features FeatureIndices, labels []bool, t Transform) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLrEstimator{}
    r.Cooccurrence       = config.Cooccurrence
    r.Kmers              = kmers
    r.Features           = features
    r.Transform          = t
    r.EpsilonLoss        = config.EpsilonLoss
    r.LogisticRegression = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, trace, &r.iterations, icv, data, labels, &r)
    if config.MaxEpochs != 0 {
      r.LogisticRegression.MaxIterations = config.MaxEpochs
    }
    return &r
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) Clone() *KmerLrOmpEstimator {
  panic("internal error")
}

func (obj *KmerLrEstimator) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) set_max_iterations(config Config) {
  if d := config.MaxEpochs - obj.iterations; config.MaxEpochs > 0 {
    if d >= 0 {
      obj.LogisticRegression.MaxIterations = d
    } else {
      obj.LogisticRegression.MaxIterations = 0
    }
  } else {
    obj.LogisticRegression.MaxIterations = int(^uint(0) >> 1)
  }
}

func (obj *KmerLrEstimator) n_params(config Config) int {
  if config.LambdaAuto != 0 {
    return config.LambdaAuto
  } else {
    if m := obj.Kmers.Len(); config.Cooccurrence {
      return CoeffIndex(m).Dim()
    } else {
      return m
    }
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) estimate(config Config, data_train []ConstVector, labels []bool) *KmerLr {
  obj.set_max_iterations(config)
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
    r.Transform                      = obj.Transform
    r.KmerLrFeatures.Binarize        = config.Binarize
    r.KmerLrFeatures.Cooccurrence    = obj   .Cooccurrence
    r.KmerLrFeatures.Features        = obj   .Features
    r.KmerLrFeatures.Kmers           = obj   .Kmers
    r.KmerLrFeatures.KmerEquivalence = config.KmerEquivalence
    return r
  }
}

func (obj *KmerLrEstimator) Estimate(config Config, data_train, data_test []ConstVector, labels []bool) *KmerLr {
  n := obj.n_params(config)
  w := [2]float64{}
  if config.Balance {
    w = compute_class_weights(labels)
  } else {
    w[0] = 1.0
    w[1] = 1.0
  }
  // create a copy of data arrays, from which to select subsets
  copy_data_train := make([]ConstVector, len(data_train))
  copy_data_test  := make([]ConstVector, len(data_test))
  for i, x := range data_train {
    copy_data_train[i] = x
  }
  for i, x := range data_test {
    copy_data_test [i] = x
  }
  s := newFeatureSelector(obj.Kmers, obj.Cooccurrence, labels, w, n)
  r := (*KmerLr)(nil)
  for {
    // select features on the initial data set
    selection, lambda, ok := s.Select(copy_data_train, obj.Theta.GetValues(), obj.Features, obj.Kmers, obj.L1Reg)
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Kmers    = selection.Kmers()
    obj.Theta    = selection.Theta()
    // create actual training and validation data sets
    selection.Data(data_train, copy_data_train)
    selection.Data(data_test , copy_data_test)

    r = obj.estimate(config, data_train, labels)
    obj.Theta.Set(r.Theta)
  }
  return r
}
