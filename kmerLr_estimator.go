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

/* -------------------------------------------------------------------------- */

type KmerLrEstimator struct {
  vectorEstimator.LogisticRegression
  KmerLrFeatures
  EpsilonLoss  float64
  reduced_data KmerDataSet
  trace        Trace
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, classifier *KmerLr, icv int) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLrEstimator{}
    r.KmerLrFeatures                    = classifier.KmerLrFeatures
    r.EpsilonLoss                       = config.EpsilonLoss
    r.LogisticRegression                = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, icv, r)
    if config.MaxIterations != 0 {
      r.LogisticRegression.MaxIterations = config.MaxIterations
    }
    if len(classifier.Theta) > 0 {
      r.SetParameters(NewDenseBareRealVector(classifier.Theta))
    }
    return r
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

func (obj *KmerLrEstimator) estimate(config Config, data KmerDataSet, transform Transform, cooccurrence bool) *KmerLr {
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
    r := &KmerLr{}
    r.Theta          = r_.(*vectorDistribution.LogisticRegression).Theta.GetValues()
    r.KmerLrFeatures = obj.KmerLrFeatures
    r.Cooccurrence   = cooccurrence
    r.Transform      = transform
    return r
  }
}

func (obj *KmerLrEstimator) estimate_loop(config Config, data KmerDataSet, transform TransformFull, lambdaAuto int, cooccurrence bool) (*KmerLr, []ConstVector) {
  if len(data.Data) == 0 {
    return nil, nil
  }
  if len(data.Kmers) != data.Data[0].Dim()-1 {
    panic("internal error")
  }
  m, n := obj.n_params(config, data.Data, lambdaAuto, cooccurrence)
  // compute class weights
  obj.LogisticRegression.SetLabels(data.Labels)
  // create a copy of data arrays, from which to select subsets
  obj.reduced_data.Data   = make([]ConstVector, len(data.Data))
  obj.reduced_data.Labels = data.Labels
  s := newFeatureSelector(config, data.Kmers, nil, cooccurrence, data.Labels, transform, obj.ClassWeights, m, n, config.EpsilonLambda)
  r := (*KmerLr)(nil)
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
    selection, lambda, ok := s.Select(data.Data, obj.Theta.GetValues(), obj.Features, obj.Kmers, nil, obj.L1Reg)
    PrintStderr(config, 1, "done\n")
    if !ok && r != nil {
      break
    }
    obj.L1Reg    = lambda
    obj.Features = selection.Features()
    obj.Kmers    = selection.Kmers()
    obj.Theta    = selection.Theta()
    // create actual training data set
    selection.Data(config, obj.reduced_data.Data, data.Data)

    PrintStderr(config, 1, "Estimating parameters with lambda=%e...\n", lambda)
    r = obj.estimate(config, obj.reduced_data, selection.Transform(), cooccurrence)
  }
  r_data := obj.reduced_data.Data
  obj.reduced_data = KmerDataSet{}
  return r, r_data
}

func (obj *KmerLrEstimator) Estimate(config Config, data KmerDataSet, transform TransformFull) []*KmerLr {
  classifiers := make([]*KmerLr, len(config.LambdaAuto))
  for i, lambda := range config.LambdaAuto {
    PrintStderr(config, 1, "Estimating classifier with %d non-zero coefficients...\n", lambda)
    classifiers[i], _ = obj.estimate_loop(config, data, transform, lambda, obj.Cooccurrence)
  }
  return classifiers
}
