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

func NewHook(config Config, trace *Trace, icv int, data []ConstVector, c []bool, estimator *KmerLrEstimator) HookType {
  loss_old := math.NaN()
  loss_new := math.NaN()
  loss := func(x ConstVector, lambda ConstScalar) float64 {
    lr := logisticRegression{}
    lr.Theta        = x.GetValues()
    lr.Lambda       = lambda.GetValue()
    lr.ClassWeights = estimator.ClassWeights
    return lr.Loss(data, c)
  }
  t := time.Now()
  s := time.Now()
  hook := func(x ConstVector, change, lambda ConstScalar, iteration int) bool {
    loss_old, loss_new = loss_new, loss_old
    n := 0
    if config.EvalLoss {
      loss_new = loss(x, lambda)
    }
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetValue() != 0.0 {
        n += 1
      }
    }
    if trace != nil {
      trace.Append(iteration, n, change.GetValue(), lambda.GetValue(), loss_new, time.Since(s))
    }
    if config.Verbose > 1 {
      if trace != nil {
        if icv != -1 {
          fmt.Printf("cv run    : %d\n", icv+1)
        }
        fmt.Printf("iteration : %d\n", iteration)
        fmt.Printf("change    : %v\n", change)
        fmt.Printf("lambda    : %v\n", lambda)
        fmt.Printf("#coef     : %d\n", n-1)
        fmt.Printf("var(#coef): %f\n", trace.CompVar(10))
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
    if trace != nil && config.EpsilonVar != 0.0 {
      if r := trace.CompVar(10); r < config.EpsilonVar {
        return true
      }
    }
    if estimator.EpsilonLoss == 0.0 {
      return false
    } else {
      return math.Abs(loss_old - loss_new) < estimator.EpsilonLoss
    }
  }
  return hook
}
