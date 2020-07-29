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
import . "github.com/pbenner/autodiff/logarithmetic"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) estimate_coordinate_loop(config Config, data_train KmerDataSet, y, w, theta0, theta1 []float64, iter int) int {
  inner_xy := make(  []float64, len(theta0))
  inner_xx := make([][]float64, len(theta0))
  norm     := make(  []float64, len(theta0))
  for j := 0; j < len(theta0); j++ {
    inner_xx[j] = make([]float64, len(theta0))
  }
  // initialize variables
  for i_, xi := range data_train.Data {
    for it := xi.ConstIterator(); it.Ok(); it.Next() {
      j1 := it.Index()
      // compute inner product between response y and feature vectors <y, x_j>
      inner_xy[j1] += w[i_]*y[i_]*it.GetConst().GetFloat64()
      // compute normalization constant
      norm[j1] += w[i_]*it.GetConst().GetFloat64()*it.GetConst().GetFloat64()
      // compute inner product between feature vectors <x_j, x_k>
      for is := xi.ConstIterator(); is.Ok(); is.Next() {
        j2 := is.Index()
        inner_xx[j1][j2] += w[i_]*it.GetConst().GetFloat64()*is.GetConst().GetFloat64()
      }
    }
  }
  for ; iter < obj.LogisticRegression.MaxIterations; iter++ {
    // coordinate descent step
    for j := 0; j < len(theta0); j++ {
      theta1_j := inner_xy[j] + norm[j]*theta1[j]
      for k := 0; k < len(theta0); k++ {
        theta1_j -= inner_xx[j][k]*theta1[k]
      }
      if j > 0 {
        // apply proximal operator
        if theta1_j >= 0.0 {
          theta1_j =  math.Max(math.Abs(theta1_j) - obj.L1Reg, 0.0)
        } else {
          theta1_j = -math.Max(math.Abs(theta1_j) - obj.L1Reg, 0.0)
        }
      }
      // normalize
      theta1_j /= (norm[j] + obj.L2Reg)
      theta0[j] = theta1[j]
      theta1[j] = theta1_j
    }
    // check convergence
    if stop, delta := obj.eval_stopping(theta0, theta1); stop {
      break
    } else {
      // execute hook if available
      if obj.LogisticRegression.Hook != nil && obj.LogisticRegression.Hook(DenseFloat64Vector(theta1), ConstFloat64(delta), ConstFloat64(obj.L1Reg), iter) {
        break
      }
    }
  }
  return iter
}

func (obj *KmerLrEstimator) estimate_coordinate(config Config, data_train KmerDataSet, transform Transform) *KmerLr {
  obj.estimate_step_size(data_train.Data)
  class_weights := compute_class_weights(data_train.Labels)
  theta0  := []float64(obj.Theta)
  theta0_ := []float64(obj.Theta)
  theta1  := []float64(obj.Theta)
  lr      := logisticRegression{theta1, obj.ClassWeights, 0.0, false, TransformFull{}, config.Pool}
  w := make([]float64, len(data_train.Data))
  z := make([]float64, len(data_train.Data))
  for iter := 0; iter < obj.LogisticRegression.MaxIterations; iter++ {
    // compute linear approximation
    for i := 0; i < len(data_train.Data); i++ {
      r   := lr.LinearPdf(data_train.Data[i].(SparseConstFloat64Vector))
      p   := math.Exp(-LogAdd(0.0, -r))
      w[i] = p*(1.0 - p)
      if data_train.Labels[i] {
        z[i] = r + (1.0 - p)/w[i]
      } else {
        z[i] = r + (0.0 - p)/w[i]
      }
      if data_train.Labels[i] {
        w[i] *= class_weights[1]
      } else {
        w[i] *= class_weights[0]
      }
    }
    // copy theta
    for j := 0; j < len(theta0); j++ {
      theta0[j] = theta1[j]
    }
    iter = obj.estimate_coordinate_loop(config, data_train, z, w, theta0_, theta1, iter)

    // check convergence
    if stop, delta := obj.eval_stopping(theta0, theta1); stop {
      break
    } else {
      // execute hook if available
      if obj.LogisticRegression.Hook != nil && obj.LogisticRegression.Hook(DenseFloat64Vector(theta1), ConstFloat64(delta), ConstFloat64(obj.L1Reg), iter) {
        break
      }
    }
  }
  obj.Theta = NewDenseFloat64Vector(theta1)
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{}
    r.Theta          = r_.(*vectorDistribution.LogisticRegression).Theta.(DenseFloat64Vector)
    r.KmerLrFeatures = obj.KmerLrFeatures
    r.Transform      = transform
    return r
  }
}
