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
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/logarithmetic"

/* -------------------------------------------------------------------------- */

type logisticRegression struct {
  Theta         []float64
  ClassWeights [2]float64
  Lambda          float64
}

/* -------------------------------------------------------------------------- */

func (obj logisticRegression) Dim() int {
  return len(obj.Theta)-1
}

func (obj logisticRegression) LogPdf(v SparseConstRealVector, gamma []float64) float64 {
  x     := v.GetSparseValues ()
  index := v.GetSparseIndices()
  // set r to first element of theta
  r := obj.Theta[0]
  // loop over x
  i := 0
  n := len(index)
  // skip first element
  if index[i] == 0 {
    i++
  }
  if gamma != nil {
    for ; i < n; i++ {
      r += float64(x[i])/gamma[index[i]]*obj.Theta[index[i]]
    }
  } else {
    for ; i < n; i++ {
      r += float64(x[i])*obj.Theta[index[i]]
    }
  }
  return -LogAdd(0.0, -r)
}

func (obj logisticRegression) ClassLogPdf(v SparseConstRealVector, gamma []float64, y bool) float64 {
  x     := v.GetSparseValues ()
  index := v.GetSparseIndices()
  // set r to first element of theta
  r := obj.Theta[0]
  // loop over x
  i := 0
  n := len(index)
  // skip first element
  if index[i] == 0 {
    i++
  }
  if gamma != nil {
    for ; i < n; i++ {
      r += float64(x[i])/gamma[index[i]]*obj.Theta[index[i]]
    }
  } else {
    for ; i < n; i++ {
      r += float64(x[i])*obj.Theta[index[i]]
    }
  }
  if y {
    return -LogAdd(0.0, -r)
  } else {
    return -LogAdd(0.0,  r)
  }
}

func (obj logisticRegression) Gradient(g []float64, data []ConstVector, labels []bool, gamma []float64) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()
  w := 0.0
  if len(g) == 0 {
    g = make([]float64, m)
  }
  if len(g) != m {
    panic("internal error")
  }
  for j, _ := range g {
    g[j] = 0
  }
  for i := 0; i < n; i++ {
    r := obj.LogPdf(data[i].(SparseConstRealVector), gamma)

    if labels[i] {
      w = obj.ClassWeights[1]*(math.Exp(r) - 1.0)
    } else {
      w = obj.ClassWeights[0]*(math.Exp(r))
    }
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      g[it.Index()] += w*it.GetValue()
    }
  }
  if obj.Lambda != 0.0 {
    for j := 1; j < m; j++ {
      if obj.Theta[j] < 0 {
        g[j] -= obj.Lambda
      } else
      if obj.Theta[j] > 0 {
        g[j] += obj.Lambda
      }
    }
  }
  return g
}

func (obj logisticRegression) Loss(data []ConstVector, c []bool, gamma []float64) float64 {
  if len(data) == 0 {
    return 0.0
  }
  n := len(data)
  m := data[0].Dim()
  r := 0.0

  for i := 0; i < n; i++ {
    r -= obj.ClassWeights[1]*obj.ClassLogPdf(data[i].(SparseConstRealVector), gamma, c[i])
  }
  if obj.Lambda != 0.0 {
    for j := 1; j < m; j++ {
      r += obj.Lambda*math.Abs(obj.Theta[j])
    }
  }
  return r
}
