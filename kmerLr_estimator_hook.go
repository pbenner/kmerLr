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

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/rprop"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

/* -------------------------------------------------------------------------- */

type HookType func(x ConstVector, change ConstScalar, epoch int) bool

/* -------------------------------------------------------------------------- */

func NewHook(config Config, trace *Trace, icv int, data []ConstVector, estimator *vectorEstimator.LogisticRegression) HookType {
  loss := func(x ConstVector) float64 {
    lr := logisticRegression{}
    lr.Theta        = x.GetValues()
    lr.Lambda       = estimator.L1Reg
    lr.ClassWeights = estimator.ClassWeights
    data, _ := estimator.GetData()
    return lr.Loss(data, nil)
  }
  hook := func(x ConstVector, change ConstScalar, epoch int) bool {
    n := 0
    l := math.NaN()
    if config.EvalLoss {
      l = loss(x)
    }
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetConst().GetValue() != 0.0 {
        n += 1
      }
    }
    if trace != nil {
      trace.Append(epoch+1, n, change.GetValue(), l)
    }
    if config.Verbose > 1 {
      if trace != nil {
        if icv != -1 {
          fmt.Printf("cv run    : %d\n", icv+1)
        }
        fmt.Printf("epoch     : %d\n", epoch+1)
        fmt.Printf("change    : %v\n", change)
        fmt.Printf("#ceof     : %d\n", n-1)
        fmt.Printf("var(#ceof): %f\n", trace.CompVar(10))
        if config.EvalLoss {
          fmt.Printf("loss      : %f\n", l)
        }
      } else {
        if icv != -1 {
          fmt.Printf("cv run: %d\n", icv+1)
        }
        fmt.Printf("epoch : %d\n", epoch)
        fmt.Printf("change: %v\n", change)
        fmt.Printf("#ceof : %d\n", n-1)
        if config.EvalLoss {
          fmt.Printf("loss  : %f\n", l)
        }
      }
      fmt.Println()
    }
    if trace != nil && config.EpsilonVar != 0.0 {
      if r := trace.CompVar(10); r < config.EpsilonVar {
        return true
      }
    }
    return false
  }
  return hook
}

func NewRpropHook(config Config, trace *Trace, icv int, data []ConstVector, estimator *KmerLrRpropEstimator) rprop.Hook {
  loss := func(x ConstVector) float64 {
    return estimator.logisticRegression.Loss(estimator.data, nil)
  }
  k    := 0
  hook := func(gradient []float64, step []float64, x ConstVector, y Scalar) bool {
    k += 1
    n := 0
    c := 0.0
    l := math.NaN()
    if config.EvalLoss {
      l = loss(x)
    }
    for i := 0; i < len(gradient); i++ {
      c += math.Abs(gradient[i])
    }
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetConst().GetValue() != 0.0 {
        n += 1
      }
    }
    if trace != nil {
      trace.Append(k, n, c, l)
    }
    if config.Verbose > 1 {
      if trace != nil {
        if icv != -1 {
          fmt.Printf("cv run    : %d\n", icv+1)
        }
        fmt.Printf("epoch     : %d\n", k)
        fmt.Printf("change    : %v\n", c)
        fmt.Printf("#ceof     : %d\n", n-1)
        fmt.Printf("var(#ceof): %f\n", trace.CompVar(10))
        if config.EvalLoss {
          fmt.Printf("loss      : %f\n", l)
        }
      } else {
        if icv != -1 {
          fmt.Printf("cv run: %d\n", icv+1)
        }
        fmt.Printf("epoch : %d\n", k)
        fmt.Printf("change: %v\n", c)
        fmt.Printf("#ceof : %d\n", n-1)
        if config.EvalLoss {
          fmt.Printf("loss  : %f\n", l)
        }
      }
      fmt.Println()
    }
    if trace != nil && config.EpsilonVar != 0.0 {
      if r := trace.CompVar(10); r < config.EpsilonVar {
        return true
      }
    }
    return false
  }
  return rprop.Hook{hook}
}
