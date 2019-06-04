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
import   "os"

import . "github.com/pbenner/ngstat/estimation"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorEstimator"

/* -------------------------------------------------------------------------- */

type HookType func(x ConstVector, change ConstScalar, epoch int) bool

/* -------------------------------------------------------------------------- */

func NewHook(config Config, trace *Trace) HookType {
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
      fmt.Printf("epoch : %d\n", epoch)
      fmt.Printf("change: %v\n", change)
      fmt.Printf("#ceof : %d\n", n)
      fmt.Println()
    }
    return false
  }
  return hook
}

/* -------------------------------------------------------------------------- */

type KmerLr struct {
  vectorEstimator.LogisticRegression
  Hook func(x ConstVector, change ConstScalar, epoch int) bool
}

/* -------------------------------------------------------------------------- */

func NewKmerLr(config Config, n int, hook HookType) *KmerLr {
  if estimator, err := vectorEstimator.NewLogisticRegression(n+1, true); err != nil {
    panic(err)
  } else {
    estimator.Hook          = hook
    estimator.Seed          = config.Seed
    estimator.L1Reg         = config.Lambda
    estimator.Epsilon       = config.Epsilon
    estimator.MaxIterations = config.MaxEpochs
    return &KmerLr{*estimator, hook}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Estimate(config Config, data []ConstVector) VectorPdf {
  if err := EstimateOnSingleTrackConstData(config.SessionConfig, &obj.LogisticRegression, data); err != nil {
    log.Fatal(err)
  }
  if r, err := obj.GetEstimate(); err != nil {
    log.Fatal(err)
  } else {
    return r
  }
  return nil
}

func (obj *KmerLr) EstimateAndTest(config Config, data_train, data_test []ConstVector) []float64 {
  // learn logistic model
  classifier := obj.Estimate(config, data_train)

  r := make([]float64, len(data_test))
  t := BareReal(0.0)
  for i, _ := range data_test {
    if err := classifier.LogPdf(&t, data_test[i]); err != nil {
      log.Fatal(err)
    }
    r[i] = t.GetValue()
  }
  return r
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
