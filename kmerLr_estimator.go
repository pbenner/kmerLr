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
  Transform    Transform
  Features     FeatureIndices
  iterations   int
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, kmers KmerClassList, trace *Trace, icv int, data []ConstVector, features FeatureIndices, labels []bool, t Transform) *KmerLrEstimator {
  if len(features) == 0 {
    features = newFeatureIndices(kmers.Len(), false)
  }
  if estimator, err := vectorEstimator.NewLogisticRegression(kmers.Len()+1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLrEstimator{}
    r.Cooccurrence       = len(kmers) != len(features)
    r.Kmers              = kmers
    r.Features           = features
    r.Transform          = t
    r.LogisticRegression = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.L1Reg          = config.Lambda
    r.LogisticRegression.AutoReg        = config.LambdaAuto
    r.LogisticRegression.Eta            = config.LambdaEta
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, trace, &r.iterations, icv, data, labels, &r.LogisticRegression)
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

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) estimate(config Config, data_train []ConstVector, labels []bool) *KmerLr {
  if err := obj.LogisticRegression.SetSparseData(data_train, labels, len(data_train)); err != nil {
    log.Fatal(err)
  }
  if err := obj.LogisticRegression.Estimate(nil, config.Pool); err != nil {
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
    r.KmerLrFeatures.KmerEquivalence = config.KmerEquivalence
    r.KmerLrFeatures.Kmers           = obj   .Kmers
    return r
  }
}

func (obj *KmerLrEstimator) estimate_prune_hook(config Config, hook_old func(x ConstVector, change, lambda ConstScalar, epoch int) bool, do_prune *bool) HookType {
  hook := func(x ConstVector, change, lambda ConstScalar, epoch int) bool {
    if r := hook_old(x, change, lambda, epoch); r {
      return true
    }
    n := 0
    m := x.Dim()
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.Index() != 0 && it.GetValue() != 0.0 {
        n += 1
      }
    }
    r := math.Round(100.0*float64(n)/float64(m))
    if int(r) < config.Prune && n >= obj.AutoReg {
      (*do_prune) = true
      return true
    }
    return false
  }
  return hook
}

func (obj *KmerLrEstimator) estimate_prune(config Config, data_train, data_test []ConstVector, labels []bool) *KmerLr {
  if config.Prune > 0 {
    var do_prune bool
    h := obj.Hook
    r := (*KmerLr)(nil)
    obj.Hook  = obj.estimate_prune_hook(config, h, &do_prune)
    for {
      do_prune = false
      obj.set_max_iterations(config)
      r = obj.estimate(config, data_train, labels)
      // check if algorithm converged
      if !do_prune {
        break
      }
      PrintStderr(config, 1, "Pruning parameter space...\n")
      r = r.Prune(data_train, data_test)
      // copy parameters
      obj.Kmers                    = r.KmerLrFeatures.Kmers
      obj.Features                 = r.KmerLrFeatures.Features
      obj.LogisticRegression.Theta = r.Theta.(DenseBareRealVector)
    }
    obj.Hook = h
    return r
  } else {
    obj.set_max_iterations(config)
    return obj.estimate(config, data_train, labels)
  }
}

func (obj *KmerLrEstimator) estimate_cooccurrence_hook(config Config, hook_old func(x ConstVector, change, lambda ConstScalar, epoch int) bool) HookType {
  hook := func(x ConstVector, change, lambda ConstScalar, epoch int) bool {
    if r := hook_old(x, change, lambda, epoch); r {
      return true
    }
    n := 0
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.Index() != 0 && it.GetValue() != 0.0 {
        n += 1
      }
    }
    if n <= config.Cooccurrence && n >= int((1.0 - 0.01)*float64(config.Cooccurrence)) {
      return true
    }
    return false
  }
  return hook
}

func (obj *KmerLrEstimator) estimate_cooccurrence(config Config, data_train, data_test []ConstVector, labels []bool) *KmerLr {
  if config.Cooccurrence > 0 && obj.Cooccurrence == false {
    h := obj.Hook
    // this hook exits the algorithm as soon as
    // the number of parameters is sufficiently reduced
    obj.Hook    = obj.estimate_cooccurrence_hook(config, h)
    // config.Cooccurrence defines the maximal number of
    // coefficients when to expand the parameter space
    obj.AutoReg = config.Cooccurrence
    r := obj.estimate_prune(config, data_train, data_test, labels)
    obj.Hook    = h
    obj.AutoReg = config.LambdaAuto
    if r.Nonzero() > config.Cooccurrence {
      // somehow the goal of reducing the parameter space was not
      // achieved => exit
      return r
    }
    PrintStderr(config, 1, "Starting co-occurrence modeling...\n")
    r  = r.Prune(data_train, data_test)
    r.ExtendCooccurrence()
    obj.Cooccurrence                     = true
    obj.Features                         = r.KmerLrFeatures.Features
    obj.Kmers                            = r.KmerLrFeatures.Kmers
    obj.LogisticRegression.Theta         = r.Theta.(DenseBareRealVector)
    obj.LogisticRegression.MaxIterations = config.MaxEpochs
    extend_counts_cooccurrence(config, data_train)
    extend_counts_cooccurrence(config, data_test)
  }
  return obj.estimate_prune(config, data_train, data_test, labels)
}

func (obj *KmerLrEstimator) Estimate(config Config, data_train, data_test []ConstVector, labels []bool) *KmerLr {
  // always prune data in case it is required for testing the classifier
  return obj.estimate_cooccurrence(config, data_train, data_test, labels).Prune(data_train, data_test)
}
