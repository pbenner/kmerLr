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

func (obj logisticRegression) Gradient(data []ConstVector, gamma []float64) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()
  w := 0.0
  g := make([]float64, m-2)

  for i := 0; i < n; i++ {
    r := obj.LogPdf(data[i].ConstSlice(0, m-1).(SparseConstRealVector), gamma)

    if _, v := data[i].(SparseConstRealVector).Last(); v == 1 {
      w = obj.ClassWeights[1]*(math.Exp(r) - 1.0)
    } else {
      w = obj.ClassWeights[0]*(math.Exp(r))
    }
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      if j := it.Index(); j != 0 && j != m-1 {
        g[j-1] += w*it.GetConst().GetValue()
      }
    }
  }
  return g
}

func (obj logisticRegression) Loss(data []ConstVector, gamma []float64, lambda float64) float64 {
  if len(data) == 0 {
    return 0.0
  }
  n := len(data)
  m := data[0].Dim()
  r := 0.0

  for i := 0; i < n; i++ {
    _, v := data[i].(SparseConstRealVector).Last()
       c := v == 1.0
    switch c {
    case true : r -= obj.ClassWeights[1]*obj.ClassLogPdf(data[i].ConstSlice(0, m-1).(SparseConstRealVector), gamma, c)
    case false: r -= obj.ClassWeights[0]*obj.ClassLogPdf(data[i].ConstSlice(0, m-1).(SparseConstRealVector), gamma, c)
    }
  }
  for j := 1; j < m-1; j++ {
    r += lambda*math.Abs(obj.Theta[j])
  }
  return r
}
