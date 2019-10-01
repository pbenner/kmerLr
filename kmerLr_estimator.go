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
  iterations   int
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, kmers KmerClassList, trace *Trace, icv int, data []ConstVector, labels []bool, t Transform) *KmerLrEstimator {
  n := kmers.Len() + 1
  if config.Cooccurrence == 0 {
    n = (kmers.Len()+1)*kmers.Len()/2 + 1
  }
  if estimator, err := vectorEstimator.NewLogisticRegression(n, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLrEstimator{}
    r.Cooccurrence       = config.Cooccurrence == 0
    r.Kmers              = kmers
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
    // alphabet parameters
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

func (obj *KmerLrEstimator) estimate(config Config, data []ConstVector, labels []bool) *KmerLr {
  if err := obj.LogisticRegression.SetSparseData(data, labels, len(data)); err != nil {
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
    r.KmerLrAlphabet.Binarize        = config.Binarize
    r.KmerLrAlphabet.Cooccurrence    = obj   .Cooccurrence
    r.KmerLrAlphabet.KmerEquivalence = config.KmerEquivalence
    r.KmerLrAlphabet.Kmers           = obj   .Kmers
    return r
  }
}

func (obj *KmerLrEstimator) estimate_prune(config Config, data []ConstVector, labels []bool, max_epochs int) *KmerLr {
  if config.Prune > 0 && (max_epochs == 0 || config.Prune < max_epochs) {
    r := (*KmerLr)(nil)
    i := 0
    for {
      // determine number of iterations
      if d := max_epochs - obj.iterations; max_epochs == 0 || d > config.Prune {
        obj.LogisticRegression.MaxIterations = config.Prune
      } else {
        obj.LogisticRegression.MaxIterations = d
      }
      i += obj.LogisticRegression.MaxIterations
      // compute new estimate
      r  = obj.estimate(config, data, labels)
      // check if converged
      if obj.iterations < i || (max_epochs > 0 && obj.iterations >= max_epochs) {
        return r
      }
      // sparsify parameters and data
      if r.Nonzero() >= config.LambdaAuto {
        r = r.Sparsify(data)
      }
      // copy parameters
      obj.Kmers                    = r.KmerLrAlphabet.Kmers
      obj.LogisticRegression.Theta = r.Theta.(DenseBareRealVector)
    }
    return r
  } else {
    return obj.estimate(config, data, labels)
  }
}

func (obj *KmerLrEstimator) Estimate(config Config, data []ConstVector, labels []bool) *KmerLr {
  if config.Cooccurrence > 0 && (config.MaxEpochs == 0 || config.Cooccurrence < config.MaxEpochs) {
    r := obj.estimate_prune(config, data, labels, config.Cooccurrence)
    r = r.Sparsify(data)
    r.ExtendCooccurrence()
    obj.Cooccurrence             = true
    obj.Kmers                    = r.KmerLrAlphabet.Kmers
    obj.LogisticRegression.Theta = r.Theta.(DenseBareRealVector)
    extend_counts_cooccurrence(config, data)
    return obj.estimate_prune(config, data, labels, config.MaxEpochs)
  } else {
    return obj.estimate_prune(config, data, labels, config.MaxEpochs).Sparsify(nil)
  }
}
