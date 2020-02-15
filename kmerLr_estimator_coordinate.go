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
  //inner_xx := make([][]float64, len(theta0)-1)
  for _, xi := range data_train.Data {
    for it := xi.ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 {
        inner_xy[j-1] += y[j]*it.GetValue()
      }
    }
  }
  // initialize variables
  for iter := 0; iter < obj.LogisticRegression.MaxIterations; iter++ {
  }
}

func (obj *KmerLrEstimator) estimate_coordinate(config Config, data_train KmerDataSet, transform Transform) *KmerLr {
  obj.estimate_step_size(data_train.Data)
  theta0  := obj.Theta.GetValues()
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
    obj.estimate_coordinate_loop(config, data_train, z, w, theta0, theta1)
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
