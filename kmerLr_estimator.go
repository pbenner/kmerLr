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
import   "bufio"
import   "log"
import   "math"
import   "os"

import . "github.com/pbenner/ngstat/estimation"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type HookType func(x ConstVector, change ConstScalar, epoch int) bool

/* -------------------------------------------------------------------------- */

func NewHook(config Config, trace *Trace, icv int) HookType {
  hook := func(x ConstVector, change ConstScalar, epoch int) bool {
    n := 0
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetConst().GetValue() != 0.0 {
        n += 1
      }
    }
    if trace != nil {
      trace.Append(epoch, n, change.GetValue())
    }
    if config.Verbose > 1 {
      if trace != nil {
        if icv != -1 {
          fmt.Printf("cv run    : %d\n", icv+1)
        }
        fmt.Printf("epoch     : %d\n", epoch)
        fmt.Printf("change    : %v\n", change)
        fmt.Printf("#ceof     : %d\n", n)
        fmt.Printf("var(#ceof): %f\n", trace.CompVar(10))
      } else {
        if icv != -1 {
          fmt.Printf("cv run: %d\n", icv+1)
        }
        fmt.Printf("epoch : %d\n", epoch)
        fmt.Printf("change: %v\n", change)
        fmt.Printf("#ceof : %d\n", n)
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

/* -------------------------------------------------------------------------- */

type KmerLrEstimator struct {
  vectorEstimator.LogisticRegression
  Kmers KmerClassList
  Hook  func(x ConstVector, change ConstScalar, epoch int) bool
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEstimator(config Config, kmers KmerClassList, hook HookType) *KmerLrEstimator {
  if estimator, err := vectorEstimator.NewLogisticRegression(kmers.Len()+1, true); err != nil {
    log.Fatal(err)
    return nil
  } else {
    estimator.Hook          = hook
    estimator.Seed          = config.Seed
    estimator.L1Reg         = config.Lambda
    estimator.Epsilon       = config.Epsilon
    estimator.MaxIterations = config.MaxEpochs
    // alphabet parameters
    return &KmerLrEstimator{*estimator, kmers, hook}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEstimator) Estimate(config Config, data []ConstVector) VectorPdf {
  if err := EstimateOnSingleTrackConstData(config.SessionConfig, &obj.LogisticRegression, data); err != nil {
    log.Fatal(err)
  }
  if r_, err := obj.LogisticRegression.GetEstimate(); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := &KmerLr{LogisticRegression: *r_.(*vectorDistribution.LogisticRegression)}
    r.KmerLrAlphabet.KmerLrAlphabetDef = config.KmerLrAlphabetDef
    r.KmerLrAlphabet.Kmers             = obj   .Kmers
    return r
  }
}

/* -------------------------------------------------------------------------- */

type Trace struct {
  Epoch   []int
  Nonzero []int
  Change  []float64
}

func (obj Trace) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  fmt.Fprintf(w, "%6s %12s %8s\n", "epoch", "change", "nonzero")
  for i := 0; i < obj.Length(); i++ {
    fmt.Fprintf(w, "%6d %12e %8d\n", obj.Epoch[i], obj.Change[i], obj.Nonzero[i])
  }
  return nil
}

func (obj Trace) Length() int {
  return len(obj.Epoch)
}

func (obj *Trace) Append(epoch, nonzero int, change float64) {
  obj.Epoch   = append(obj.Epoch  , epoch)
  obj.Nonzero = append(obj.Nonzero, nonzero)
  obj.Change  = append(obj.Change , change)
}

func (obj Trace) CompVar(n int) float64 {
  if len(obj.Nonzero) < n {
    return math.NaN()
  }
  v := obj.Nonzero[len(obj.Nonzero)-n:len(obj.Nonzero)]
  m := 0.0
  for i := 0; i < n; i++ {
    m += float64(v[i])
  }
  m /= float64(n)
  r := 0.0
  for i := 0; i < n; i++ {
    s := float64(v[i]) - m
    r += s*s
  }
  r /= float64(n)
  return r
}
