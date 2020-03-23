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

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type ScoresLr struct {
  ScoresLrFeatures
  Theta     []float64
  Transform   Transform
}

/* -------------------------------------------------------------------------- */

func NewScoresLr(theta []float64, features ScoresLrFeatures) *ScoresLr {
  return &ScoresLr{Theta: theta, ScoresLrFeatures: features}
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Clone() *ScoresLr {
  r := ScoresLr{}
  r.Theta = make([]float64, len(obj.Theta))
  for i := 0; i < len(obj.Theta); i++ {
    r.Theta[i] = obj.Theta[i]
  }
  r.ScoresLrFeatures = obj.ScoresLrFeatures.Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Predict(config Config, data []ConstVector) []float64 {
  lr := logisticRegression{}
  lr.Theta  = obj   .Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.PoolLR
  r := make([]float64, len(data))
  for i, _ := range data {
    r[i] = lr.LogPdf(data[i].(SparseConstRealVector))
  }
  return r
}

func (obj *ScoresLr) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj   .Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.PoolLR
  if config.Balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  return lr.Loss(data, c)
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Nonzero() int {
  n  := 0
  for i := 1; i < len(obj.Theta); i++ {
    if obj.Theta[i] != 0.0 {
      n++
    }
  }
  return n
}
