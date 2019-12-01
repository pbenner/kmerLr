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

/* -------------------------------------------------------------------------- */

type ScoresId     =    int
type ScoresIdPair = [2]int

type ScoresLrCoefficientsSet struct {
  Offset         float64
  Coefficients []float64
  Index          map[ScoresId    ]int
  IndexPairs     map[ScoresIdPair]int
}

func NewScoresLrCoefficientsSet() *ScoresLrCoefficientsSet {
  r := ScoresLrCoefficientsSet{}
  r.Index      = make(map[ScoresId]int)
  r.IndexPairs = make(map[ScoresIdPair]int)
  return &r
}

func (obj *ScoresLrCoefficientsSet) Add(id int, value float64) {
  obj.Set(id, obj.Get(id) + value)
}

func (obj *ScoresLrCoefficientsSet) AddPair(id1, id2 int, value float64) {
  obj.SetPair(id1, id2, obj.GetPair(id1, id2) + value)
}

func (obj *ScoresLrCoefficientsSet) DivAll(value float64) {
  obj.Offset /= value
  for i, _ := range obj.Coefficients {
    obj.Coefficients[i] /= value
  }
}

func (obj *ScoresLrCoefficientsSet) Set(id int, value float64) {
  if i, ok := obj.Index[id]; ok {
    obj.Coefficients[i] = value
  } else {
    if value != 0.0 {
      obj.Index[id]    = len   (obj.Coefficients)
      obj.Coefficients = append(obj.Coefficients, value)
    }
  }
}

func (obj *ScoresLrCoefficientsSet) SetPair(id1, id2 int, value float64) {
  pair := ScoresIdPair{id1, id2}
  if i, ok := obj.IndexPairs[pair]; ok {
    obj.Coefficients[i] = value
  } else {
    if value != 0.0 {
      obj.IndexPairs[pair] = len   (obj.Coefficients)
      obj.Coefficients     = append(obj.Coefficients, value)
    }
  }
}

func (obj *ScoresLrCoefficientsSet) Get(id int) float64 {
  if i, ok := obj.Index[id]; ok {
    return obj.Coefficients[i]
  } else {
    return 0.0
  }
}

func (obj *ScoresLrCoefficientsSet) GetPair(id1, id2 int) float64 {
  pair := ScoresIdPair{id1, id2}
  if i, ok := obj.IndexPairs[pair]; ok {
    return obj.Coefficients[i]
  } else {
    return 0.0
  }
}

func (obj *ScoresLrCoefficientsSet) AddCoefficients(b *ScoresLrCoefficientsSet) {
  obj.Offset += b.Offset
  for id, i := range b.Index {
    obj.Add(id, b.Coefficients[i])
  }
  for pair, i := range b.IndexPairs {
    obj.AddPair(pair[0], pair[1], b.Coefficients[i])
  }
}

func (obj *ScoresLrCoefficientsSet) MaxCoefficients(b *ScoresLrCoefficientsSet) {
  if math.Abs(obj.Offset) < math.Abs(b.Offset) {
    obj.Offset = b.Offset
  }
  for id, i := range b.Index {
    if v := obj.Get(id); math.Abs(v) < math.Abs(b.Coefficients[i]) {
      obj.Set(id, b.Coefficients[i])
    }
  }
  for pair, i := range b.IndexPairs {
    if v := obj.GetPair(pair[0], pair[1]); math.Abs(v) < math.Abs(b.Coefficients[i]) {
      obj.SetPair(pair[0], pair[1], b.Coefficients[i])
    }
  }
}

func (obj *ScoresLrCoefficientsSet) MinCoefficients(b *ScoresLrCoefficientsSet) {
  if math.Abs(obj.Offset) > math.Abs(b.Offset) {
    obj.Offset = b.Offset
  }
  for id, i := range obj.Index {
    if v := b.Get(id); math.Abs(obj.Coefficients[i]) > math.Abs(v) {
      obj.Set(id, v)
    }
  }
  for pair, i := range obj.IndexPairs {
    if v := b.GetPair(pair[0], pair[1]); math.Abs(obj.Coefficients[i]) > math.Abs(v) {
      obj.SetPair(pair[0], pair[1], v)
    }
  }
}

func (obj *ScoresLrCoefficientsSet) AsScoresLr(features ScoresLrFeatures) *ScoresLr {
  f := FeatureIndices{}
  v := NullDenseBareRealVector(1)
  v[0] = ConstReal(obj.Offset)
  for id, i := range obj.Index {
    f = append(f, [2]int{id, id})
    v = append(v, ConstReal(obj.Coefficients[i]))
  }
  for pair, i := range obj.IndexPairs {
    f = append(f, [2]int{pair[0], pair[1]})
    v = append(v, ConstReal(obj.Coefficients[i]))
  }
  features          = features.Clone()
  features.Features = f
  return NewScoresLr(v, features)
}
