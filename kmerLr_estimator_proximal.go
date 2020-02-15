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
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

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

func (obj *KmerLrEstimator) estimate_step_size(x []ConstVector) {
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
  obj.SetStepSize(stepSize)
}

func (obj *KmerLrEstimator) estimate_proximal(config Config, data_train KmerDataSet, transform Transform) *KmerLr {
  obj.estimate_step_size(data_train.Data)
  theta0 := obj.Theta.GetValues()
  theta1 := obj.Theta.GetValues()
  lr     := logisticRegression{theta1, obj.ClassWeights, 0.0, false, TransformFull{}, config.Pool}
  for i := 0; i < obj.LogisticRegression.MaxIterations; i++ {
    // receive step size during each iteration, since the hook
    // might modify it
    s := obj.GetStepSize()
    g := lr .Gradient(nil, data_train.Data, data_train.Labels)
    for k := 0; k < len(theta1); k++ {
      theta0[k] = theta1[k]
      theta1[k] = theta1[k] - s*g[k]
      if k > 0 {
        if theta1[k] >= 0.0 {
          theta1[k] =  math.Max(math.Abs(theta1[k]) - s*obj.L1Reg, 0.0)
        } else {
          theta1[k] = -math.Max(math.Abs(theta1[k]) - s*obj.L1Reg, 0.0)
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
