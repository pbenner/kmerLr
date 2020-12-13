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

import   "fmt"
import   "math"
import   "time"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type HookType func(x ConstVector, change, lambda ConstScalar, epoch int) bool

/* -------------------------------------------------------------------------- */

func NewHook(config Config, icv int, estimator *KmerLrEstimator) HookType {
  loss_old := math.NaN()
  loss_new := math.NaN()
  loss := func(x ConstVector, lambda ConstScalar) float64 {
    lr := logisticRegression{}
    lr.Theta        = x.(DenseFloat64Vector)
    lr.Lambda       = lambda.GetFloat64()
    lr.ClassWeights = estimator.ClassWeights
    return lr.Loss(estimator.reduced_data.Data, estimator.reduced_data.Labels)
  }
  t := time.Now()
  s := time.Now()
  c := 0
  hook := func(x ConstVector, change, lambda ConstScalar, iteration int) bool {
    loss_old, loss_new = loss_new, loss_old
    n := 0
    if config.EvalLoss {
      loss_new = loss(x, lambda)
      if config.AdaptStepSize {
        if loss_new > loss_old {
          PrintStderr(config, 2, "Warning: optimization algorithm is oscillating, decresing step size...\n")
          gamma := estimator.GetStepSize()
          estimator.SetStepSize(gamma/2.0)
          c  = 0
        } else {
          c += 1
        }
        if c >= 10 {
          PrintStderr(config, 2, "Warning: optimization algorithm converges slowly, increasing step size...\n")
          gamma := estimator.GetStepSize()
          estimator.SetStepSize(gamma*2.0)
          c = 0
        }
      }
    }
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetConst().GetFloat64() != 0.0 {
        n += 1
      }
    }
    if config.SaveTrace {
      estimator.trace.Append(iteration, n, change.GetFloat64(), lambda.GetFloat64(), loss_new, time.Since(s))
    }
    if config.Verbose > 1 {
      if config.SaveTrace {
        if icv != -1 {
          fmt.Printf("cv run    : %d\n", icv+1)
        }
        fmt.Printf("iteration : %d\n", iteration)
        fmt.Printf("change    : %v\n", change)
        fmt.Printf("lambda    : %v\n", lambda)
        fmt.Printf("#coef     : %d\n", n-1)
        fmt.Printf("var(#coef): %f\n", estimator.trace.CompVar(10))
        if config.EvalLoss {
          fmt.Printf("loss      : %f\n", loss_new)
        }
        fmt.Printf("time      : %v\n", time.Since(t))
      } else {
        if icv != -1 {
          fmt.Printf("cv run: %d\n", icv+1)
        }
        fmt.Printf("iteration: %d\n", iteration)
        fmt.Printf("change   : %v\n", change)
        fmt.Printf("lambda   : %v\n", lambda)
        fmt.Printf("#coef    : %d\n", n-1)
        if config.EvalLoss {
          fmt.Printf("loss     : %f\n", loss_new)
        }
        fmt.Printf("time     : %v\n", time.Since(t))
      }
      fmt.Println()
    }
    t = time.Now()
    if estimator.EpsilonLoss == 0.0 {
      return false
    } else {
      return math.Abs(loss_old - loss_new) < estimator.EpsilonLoss
    }
  }
  return hook
}
