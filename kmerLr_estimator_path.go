/* Copyright (C) 2020 Philipp Benner
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
import   "math"
import   "os"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerRegularizationPath struct {
  Estimator []int
  Lambda    []float64
  Norm      []float64
  Theta   [][]float64
  Kmers     KmerClassList
  Index     map[KmerClassId]int
}

/* -------------------------------------------------------------------------- */

func (obj KmerRegularizationPath) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  // print header
  if len(obj.Estimator) > 0 {
    fmt.Fprintf(w, "%9s ", "estimator")
  }
  fmt.Fprintf(w, "%13s %13s %s\n", "lambda", "norm", "theta")

  // print values
  for i := 0; i < obj.Length(); i++ {
    if len(obj.Estimator) > 0 {
      fmt.Fprintf(w, "%9d ", obj.Estimator[i])
    }
    fmt.Fprintf(w, "%13e %13e", obj.Lambda[i], obj.Norm[i])
    for j := 0; j < len(obj.Theta[i]); j++ {
      if j == 0 {
        fmt.Fprintf(w, " %e", obj.Theta[i][j])
      } else {
        fmt.Fprintf(w, ",%e", obj.Theta[i][j])
      }
    }
    fmt.Fprintf(w, "\n")
  }
  return nil
}

func (obj *KmerRegularizationPath) Length() int {
  return len(obj.Lambda)
}

func (obj *KmerRegularizationPath) Append(estimator int, lambda float64, kmers KmerClassList, theta []float64) {
  if len(obj.Index) == 0 {
    obj.Index = make(map[KmerClassId]int)
  }
  v := 0.0
  t := make([]float64, len(obj.Index))
  for j := 1; j < len(theta); j++ {
    if theta[j] == 0.0 {
      continue
    }
    v += math.Abs(theta[j])
    if k, ok := obj.Index[kmers[j-1].KmerClassId]; !ok {
      k = len(obj.Index)
      obj.Index[kmers[j-1].KmerClassId] = k
      obj.Kmers = append(obj.Kmers, kmers[j-1])
      t = append(t, theta[j])
    } else {
      t[k] = theta[j]
    }
  }
  if estimator != -1 {
    obj.Estimator = append(obj.Estimator, estimator)
  }
  obj.Lambda = append(obj.Lambda, lambda)
  obj.Norm   = append(obj.Norm  , v)
  obj.Theta  = append(obj.Theta , t)
  // augment previous entries with zeroes
  for i := 0; i < len(obj.Theta); i++ {
    for j := len(obj.Theta[i]); j < len(t); j++ {
      obj.Theta[i] = append(obj.Theta[i], 0.0)
    }
  }
}

func (obj *KmerRegularizationPath) AppendPath(estimator int, path KmerRegularizationPath) {
  for i := 0; i < path.Length(); i++ {
    obj.Append(estimator, path.Lambda[i], path.Kmers, path.Theta[i])
  }
}
