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

/* -------------------------------------------------------------------------- */

type KmerLrEstimator struct {
  vectorEstimator.LogisticRegression
  KmerLrFeatures
  EpsilonLoss  float64
  // reduced data sets
  reduced_data_train KmerDataSet
  reduced_data_test  KmerDataSet
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, classifier *KmerLr, trace *Trace, icv int) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLrEstimator{}
    r.KmerLrFeatures                    = classifier.KmerLrFeatures
    r.EpsilonLoss                       = config.EpsilonLoss
    r.LogisticRegression                = *estimator
    r.LogisticRegression.Balance        = config.Balance
    r.LogisticRegression.Seed           = config.Seed
    r.LogisticRegression.Epsilon        = config.Epsilon
    r.LogisticRegression.StepSizeFactor = config.StepSizeFactor
    r.LogisticRegression.Hook           = NewHook(config, trace, icv, &r)
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

func (obj *KmerLrEstimator) eval_stopping(xs, x1 []float64) (bool, float64) {
  // evaluate stopping criterion
  max_x     := 0.0
  max_delta := 0.0
  delta     := 0.0
  for i := 0; i < len(xs); i++ {
    if math.IsNaN(x1[i]) {
      return true, math.NaN()
    }
    max_x     = math.Max(max_x    , math.Abs(x1[i]))
    max_delta = math.Max(max_delta, math.Abs(x1[i] - xs[i]))
  }
  if max_x != 0.0 {
    delta = max_delta/max_x
  } else {
    delta = max_delta
  }
  if max_x != 0.0 && max_delta/max_x <= obj.LogisticRegression.Epsilon ||
    (max_x == 0.0 && max_delta == 0.0) {
    return true, delta
  }
  return false, delta
}

func (obj *KmerLrEstimator) estimate_step_size(x []ConstVector) float64 {
  max_squared_sum := 0.0
  max_weight      := 1.0
  for _, x := range x {
    r  := 0.0
    it := x.ConstIterator()
    // skip first element
    if it.Ok() {
      it.Next()
    }
    for ; it.Ok(); it.Next() {
      r += it.GetValue()*it.GetValue()
    }
    if r > max_squared_sum {
      max_squared_sum = r
    }
  }
  L := (0.25*(max_squared_sum + 1.0) + obj.L2Reg/float64(len(x)))
  L *= max_weight
  stepSize := 1.0/(2.0*L + math.Min(2.0*obj.L2Reg, L))
  stepSize *= obj.StepSizeFactor
  return stepSize
}

func (obj *KmerLrEstimator) estimate_debug(config Config, data_train KmerDataSet, transform Transform) *KmerLr {
  theta0 := obj.Theta.GetValues()
  theta1 := obj.Theta.GetValues()
  gamma  := obj.estimate_step_size(data_train.Data)
  lr     := logisticRegression{theta1, obj.ClassWeights, 0.0, false, TransformFull{}, config.Pool}
  for i := 0; i < obj.LogisticRegression.MaxIterations; i++ {
    g := lr.Gradient(nil, data_train.Data, data_train.Labels)
    for k := 0; k < len(theta1); k++ {
      theta0[k] = theta1[k]
      theta1[k] = theta1[k] - gamma*g[k]
      if k > 0 {
        if theta1[k] >= 0.0 {
          theta1[k] =  math.Max(math.Abs(theta1[k]) - gamma*obj.L1Reg, 0.0)
        } else {
          theta1[k] = -math.Max(math.Abs(theta1[k]) - gamma*obj.L1Reg, 0.0)
        }
      }
    }
    // check convergence
    if stop, delta := obj.eval_stopping(theta0, theta1); stop {
      break
    } else {
      // execute hook if available
      if obj.LogisticRegression.Hook != nil && obj.LogisticRegression.Hook(DenseConstRealVector(theta1), ConstReal(delta), ConstReal(obj.L1Reg), i) {
        break
      }
    }
  }
  obj.Theta = NewDenseBareRealVector(theta1)
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{}
    r.Theta          = r_.(*vectorDistribution.LogisticRegression).Theta.GetValues()
    r.KmerLrFeatures = obj.KmerLrFeatures
    r.Transform      = transform
    return r
  }
}

func (obj *KmerLrEstimator) estimate(config Config, data_train KmerDataSet, transform Transform) *KmerLr {
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
    r.Theta          = r_.(*vectorDistribution.LogisticRegression).Theta.GetValues()
    r.KmerLrFeatures = obj.KmerLrFeatures
    r.Transform      = transform
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
    r = obj.estimate(config, obj.reduced_data_train, selection.Transform())
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
