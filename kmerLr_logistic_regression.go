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
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type logisticRegression struct {
  Theta         []float64
  ClassWeights [2]float64
  Lambda          float64
  Cooccurrence    bool
  // apply transform on the fly because training data
  // does not include co-occurrences
  Transform       TransformFull
  Pool            threadpool.ThreadPool
}

/* -------------------------------------------------------------------------- */

func (obj logisticRegression) Dim() int {
  return len(obj.Theta)-1
}

func (obj logisticRegression) LinearPdf(x SparseConstFloat64Vector) float64 {
  i := x.GetSparseIndices()
  v := x.GetSparseValues ()
  n := x.Dim()-1
  q := len(i)
  r := obj.Theta[0]
  if i[0] != 0 {
    panic("internal error")
  }
  if obj.Cooccurrence {
    if len(obj.Theta) != CoeffIndex(n).Dim() {
      panic("internal error")
    }
  } else {
    if len(obj.Theta) != n+1 {
      panic("internal error")
    }
  }
  if obj.Transform.Nil() {
    for j := 1; j < q; j++ {
      r += v[j]*obj.Theta[i[j]]
    }
    if obj.Cooccurrence {
      s := make([]float64, q)
      obj.Pool.RangeJob_(1, q, func(jFrom, jTo int, pool threadpool.ThreadPool, erf func() error) error {
        for j1 := jFrom; j1 < jTo; j1++ {
          for j2 := j1+1; j2 < q; j2++ {
            i1 := i[j1]-1
            i2 := i[j2]-1
            j  := CoeffIndex(n).Ind2Sub(i1, i2)
            s[j1] += v[j1]*v[j2]*obj.Theta[j]
          }
        }
        return nil
      })
      for i := 1; i < q; i++ {
        r += s[i]
      }
    }
  } else {
    for j := 1; j < q; j++ {
      r += obj.Transform.Apply(v[j], i[j])*obj.Theta[i[j]]
    }
    if obj.Cooccurrence {
      s := make([]float64, q)
      obj.Pool.RangeJob_(1, q, func(jFrom, jTo int, pool threadpool.ThreadPool, erf func() error) error {
        for j1 := jFrom; j1 < jTo; j1++ {
          for j2 := j1+1; j2 < q; j2++ {
            i1 := i[j1]-1
            i2 := i[j2]-1
            j  := CoeffIndex(n).Ind2Sub(i1, i2)
            s[j1] += obj.Transform.Apply(v[j1]*v[j2], j)*obj.Theta[j]
          }
        }
        return nil
      })
      for i := 1; i < q; i++ {
        r += s[i]
      }
    }
  }
  return r
}

func (obj logisticRegression) ClassLogPdf(x SparseConstFloat64Vector, y bool) float64 {
  r := obj.LinearPdf(x)
  if y {
    return -LogAdd(0.0, -r)
  } else {
    return -LogAdd(0.0,  r)
  }
}

func (obj logisticRegression) LogPdf(v SparseConstFloat64Vector) float64 {
  return obj.ClassLogPdf(v, true)
}

func (obj logisticRegression) Gradient(g []float64, data []ConstVector, labels []bool) []float64 {
  if len(data) == 0 {
    return nil
  }
  if len(g) == 0 {
    g = make([]float64, len(obj.Theta))
  } else {
    if len(g) != len(obj.Theta) {
      panic("internal error")
    }
    // initialize gradient
    for j, _ := range g {
      g[j] = 0
    }
  }
  for i_ := 0; i_ < len(data); i_++ {
    w := 0.0
    r := obj.LogPdf(data[i_].(SparseConstFloat64Vector))
    i := data[i_].(SparseConstFloat64Vector).GetSparseIndices()
    v := data[i_].(SparseConstFloat64Vector).GetSparseValues ()
    n := data[i_].Dim()-1
    q := len(i)

    if labels[i_] {
      w = obj.ClassWeights[1]*(math.Exp(r) - 1.0)
    } else {
      w = obj.ClassWeights[0]*(math.Exp(r))
    }
    if obj.Transform.Nil() {
      for j := 0; j < q; j++ {
        g[i[j]] += w*v[j]
      }
      if obj.Cooccurrence {
        obj.Pool.RangeJob_(1, q, func(jFrom, jTo int, pool threadpool.ThreadPool, erf func() error) error {
          for j1 := jFrom; j1 < jTo; j1++ {
            for j2 := j1+1; j2 < q; j2++ {
              i1 := i[j1]-1
              i2 := i[j2]-1
              j  := CoeffIndex(n).Ind2Sub(i1, i2)
              g[j] += w*v[j1]*v[j2]
            }
          }
          return nil
        })
      }
    } else {
      for j := 0; j < q; j++ {
        g[i[j]] += w*obj.Transform.Apply(v[j], i[j])
      }
      if obj.Cooccurrence {
        obj.Pool.RangeJob_(1, q, func(jFrom, jTo int, pool threadpool.ThreadPool, erf func() error) error {
          for j1 := jFrom; j1 < jTo; j1++ {
            for j2 := j1+1; j2 < q; j2++ {
              i1 := i[j1]-1
              i2 := i[j2]-1
              j  := CoeffIndex(n).Ind2Sub(i1, i2)
              g[j] += w*obj.Transform.Apply(v[j1]*v[j2], j)
            }
          }
          return nil
        })
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

func (obj logisticRegression) Loss(data []ConstVector, c []bool) float64 {
  if len(data) == 0 {
    return 0.0
  }
  n := len(data)
  m := data[0].Dim()
  r := 0.0

  for i := 0; i < n; i++ {
    if c[i] {
      r -= obj.ClassWeights[1]*obj.ClassLogPdf(data[i].(SparseConstFloat64Vector), c[i])
    } else {
      r -= obj.ClassWeights[0]*obj.ClassLogPdf(data[i].(SparseConstFloat64Vector), c[i])
    }
  }
  if obj.Lambda != 0.0 {
    for j := 1; j < m; j++ {
      r += obj.Lambda*math.Abs(obj.Theta[j])
    }
  }
  return r
}
