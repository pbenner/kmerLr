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
  Cooccurrence    bool
}

/* -------------------------------------------------------------------------- */

func (obj logisticRegression) Dim() int {
  return len(obj.Theta)-1
}

func (obj logisticRegression) ClassLogPdf(x SparseConstRealVector, gamma []float64, y bool) float64 {
  i := x.GetSparseIndices()
  v := x.GetSparseValues ()
  n := x.Dim()-1
  q := len(i)
  r := obj.Theta[0]
  if i[0] != 0 {
    panic("internal error")
  }
  if gamma != nil {
    for j := 1; j < q; j++ {
      r += float64(v[j])/gamma[i[j]]*obj.Theta[i[j]]
    }
    if obj.Cooccurrence {
      for j1 := 1; j1 < q; j1++ {
        for j2 := j1+1; j2 < q; j2++ {
          i1 := i[j1]-1
          i2 := i[j2]-1
          j  := CoeffIndex(n).Ind2Sub(i1, i2)
          r += float64(v[j1]*v[j2])/gamma[j]*obj.Theta[j]
        }
      }
    }
  } else {
    for j := 1; j < q; j++ {
      r += float64(v[j])*obj.Theta[i[j]]
    }
    if obj.Cooccurrence {
      for j1 := 1; j1 < q; j1++ {
        for j2 := j1+1; j2 < q; j2++ {
          i1 := i[j1]-1
          i2 := i[j2]-1
          j  := CoeffIndex(n).Ind2Sub(i1, i2)
          r += float64(v[j1]*v[j2])*obj.Theta[j]
        }
      }
    }
  }
  if y {
    return -LogAdd(0.0, -r)
  } else {
    return -LogAdd(0.0,  r)
  }
}

func (obj logisticRegression) LogPdf(v SparseConstRealVector, gamma []float64) float64 {
  return obj.ClassLogPdf(v, gamma, true)
}

func (obj logisticRegression) Gradient(g []float64, data []ConstVector, labels []bool, gamma []float64) []float64 {
  if len(data) == 0 {
    return nil
  }
  n := len(data)
  m := data[0].Dim()-1
  if len(g) == 0 {
    if obj.Cooccurrence {
      g = make([]float64, CoeffIndex(m).Dim())
    } else {
      g = make([]float64, m+1)
    }
  } else {
    if obj.Cooccurrence {
      if len(g) != CoeffIndex(m).Dim() {
        panic("internal error")
      }
    } else {
      if len(g) != m+1 {
        panic("internal error")
      }
    }
    // initialize gradient
    for j, _ := range g {
      g[j] = 0
    }
  }
  for i := 0; i < n; i++ {
    w := 0.0
    r := obj.LogPdf(data[i].(SparseConstRealVector), gamma)

    if labels[i] {
      w = obj.ClassWeights[1]*(math.Exp(r) - 1.0)
    } else {
      w = obj.ClassWeights[0]*(math.Exp(r))
    }
    for it := data[i].ConstIterator(); it.Ok(); it.Next() {
      g[it.Index()] += w*it.GetValue()
    }
    if obj.Cooccurrence {
      for it1 := data[i].ConstIterator(); it1.Ok(); it1.Next() {
        it2 := it1.CloneConstIterator()
        it2.Next()
        for ; it2.Ok(); it2.Next() {
          i1 := it1.Index()-1
          i2 := it2.Index()-1
          j  := CoeffIndex(m).Ind2Sub(i1, i2)
          g[j] += w*it1.GetValue()*it2.GetValue()
        }
      }
    }
  }
  if obj.Lambda != 0.0 {
    for j := 1; j < len(obj.Theta); j++ {
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
