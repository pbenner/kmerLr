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

func (obj *KmerLrEstimator) estimate_coordinate_loop(config Config, data_train KmerDataSet, y, w, theta0, theta1 []float64) {
  inner_xy := make(  []float64, len(theta0)-1)
  inner_xx := make([][]float64, len(theta0)-1)
  norm     := make(  []float64, len(theta0))
  for j := 0; j < len(theta0)-1; j++ {
    inner_xx[j] = make([]float64, len(theta0)-1)
  }
  // initialize variables
  for i_, xi := range data_train.Data {
    for it := xi.ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 {
        // compute inner product between response y and feature vectors <y, x_j>
        inner_xy[j-1] += w[i_]*y[i_]*it.GetValue()
        // compute normalization constant
        norm[j] += w[i_]*it.GetValue()*it.GetValue()
      } else {
        // compute normalization constant for offset parameter theta_0
        norm[j] += w[i_]
      }
      // compute inner product between feature vectors <x_j, x_k>
      for is := xi.ConstIterator(); is.Ok(); is.Next() {
        if j1, j2 := it.Index(), is.Index(); j1 != 0 && j2 != 0 {
          inner_xx[j1-1][j2-1] += w[i_]*it.GetValue()*is.GetValue()
        }
      }
    }
  }
  for iter := 0; iter < obj.LogisticRegression.MaxIterations; iter++ {
    // coordinate descent step
    for j := 1; j < len(theta0); j++ {
      theta1_j := norm[j]*theta1[j] + inner_xy[j-1]
      for k := 1; k < len(theta0); k++ {
        theta1_j -= inner_xx[j-1][k-1]*theta1[k]
      }
      // apply proximal operator
      if theta1_j >= 0.0 {
        theta1_j =  math.Max(math.Abs(theta1_j) - obj.L1Reg, 0.0)
      } else {
        theta1_j = -math.Max(math.Abs(theta1_j) - obj.L1Reg, 0.0)
      }
      // normalize
      theta1_j /= norm[j]
      theta0[j] = theta1[j]
      theta1[j] = theta1_j
    }
    // check convergence
    if stop, _ := obj.eval_stopping(theta0, theta1); stop {
      break
    }
  }
}

func (obj *KmerLrEstimator) estimate_coordinate(config Config, data_train KmerDataSet, transform Transform) *KmerLr {
  obj.estimate_step_size(data_train.Data)
  theta0  := obj.Theta.GetValues()
  theta0_ := obj.Theta.GetValues()
  theta1  := obj.Theta.GetValues()
  lr      := logisticRegression{theta1, obj.ClassWeights, 0.0, false, TransformFull{}, config.Pool}
  w := make([]float64, len(data_train.Data))
  z := make([]float64, len(data_train.Data))
  for epoch := 0; epoch < obj.LogisticRegression.MaxIterations; epoch++ {
    // compute linear approximation
    for i := 0; i < len(data_train.Data); i++ {
      r   := lr.LinearPdf(data_train.Data[i].(SparseConstRealVector))
      p   := math.Exp(-LogAdd(0.0, -r))
      w[i] = p*(1-p)
      if data_train.Labels[i] {
        z[i] = r + (1.0 - p)/w[i]
      } else {
        z[i] = r + (0.0 - p)/w[i]
      }
    }
    // copy theta
    for j := 0; j < len(theta0); j++ {
      theta0[j] = theta1[j]
    }
    obj.estimate_coordinate_loop(config, data_train, z, w, theta0_, theta1)

    // check convergence
    if stop, delta := obj.eval_stopping(theta0, theta1); stop {
      break
    } else {
      // execute hook if available
      if obj.LogisticRegression.Hook != nil && obj.LogisticRegression.Hook(DenseConstRealVector(theta1), ConstReal(delta), ConstReal(obj.L1Reg), epoch) {
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
