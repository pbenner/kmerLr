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
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerClassIdPair struct {
  Kmer1 KmerClassId
  Kmer2 KmerClassId
}

func NewKmerClassIdPair(kmer1, kmer2 KmerClass) KmerClassIdPair {
  if kmer1.Less(kmer2) {
    return KmerClassIdPair{kmer1.KmerClassId, kmer2.KmerClassId}
  } else {
    return KmerClassIdPair{kmer2.KmerClassId, kmer1.KmerClassId}
  }
}

type KmerLrCoefficientsSet struct {
  Offset         float64
  Coefficients []float64
  Kmers          KmerClassSet
  Index          map[KmerClassId    ]int
  IndexPairs     map[KmerClassIdPair]int
}

func NewKmerLrCoefficientsSet() *KmerLrCoefficientsSet {
  r := KmerLrCoefficientsSet{}
  r.Kmers      = make(KmerClassSet)
  r.Index      = make(map[KmerClassId]int)
  r.IndexPairs = make(map[KmerClassIdPair]int)
  return &r
}

func (obj *KmerLrCoefficientsSet) Add(kmer KmerClass, value float64) {
  obj.Set(kmer, obj.Get(kmer) + value)
}

func (obj *KmerLrCoefficientsSet) AddPair(kmer1, kmer2 KmerClass, value float64) {
  obj.SetPair(kmer1, kmer2, obj.GetPair(kmer1, kmer2) + value)
}

func (obj *KmerLrCoefficientsSet) DivAll(value float64) {
  obj.Offset /= value
  for i, _ := range obj.Coefficients {
    obj.Coefficients[i] /= value
  }
}

func (obj *KmerLrCoefficientsSet) Set(kmer KmerClass, value float64) {
  if i, ok := obj.Index[kmer.KmerClassId]; ok {
    obj.Coefficients[i] = value
  } else {
    if value != 0.0 {
      obj.Kmers[kmer.KmerClassId] = kmer.Elements
      obj.Index[kmer.KmerClassId] = len(obj.Coefficients)
      obj.Coefficients = append(obj.Coefficients, value)
    }
  }
}

func (obj *KmerLrCoefficientsSet) SetPair(kmer1, kmer2 KmerClass, value float64) {
  pair := NewKmerClassIdPair(kmer1, kmer2)
  if i, ok := obj.IndexPairs[pair]; ok {
    obj.Coefficients[i] = value
  } else {
    if value != 0.0 {
      if _, ok := obj.Index[kmer1.KmerClassId]; !ok {
        obj.Kmers[kmer1.KmerClassId] = kmer1.Elements
        obj.Index[kmer1.KmerClassId] = len(obj.Coefficients)
        obj.Coefficients = append(obj.Coefficients, 0.0)
      }
      if _, ok := obj.Index[kmer2.KmerClassId]; !ok {
        obj.Kmers[kmer2.KmerClassId] = kmer2.Elements
        obj.Index[kmer2.KmerClassId] = len(obj.Coefficients)
        obj.Coefficients = append(obj.Coefficients, 0.0)
      }
      obj.IndexPairs[pair] = len(obj.Coefficients)
      obj.Coefficients     = append(obj.Coefficients, value)
    }
  }
}

func (obj *KmerLrCoefficientsSet) Get(kmer KmerClass) float64 {
  if i, ok := obj.Index[kmer.KmerClassId]; ok {
    return obj.Coefficients[i]
  } else {
    return 0.0
  }
}

func (obj *KmerLrCoefficientsSet) GetPair(kmer1, kmer2 KmerClass) float64 {
  pair := NewKmerClassIdPair(kmer1, kmer2)
  if i, ok := obj.IndexPairs[pair]; ok {
    return obj.Coefficients[i]
  } else {
    return 0.0
  }
}

func (obj *KmerLrCoefficientsSet) AddCoefficients(b *KmerLrCoefficientsSet) {
  obj.Offset += b.Offset
  for id, i := range b.Index {
    kmer := KmerClass{id, b.Kmers[id]}
    obj.Add(kmer, b.Coefficients[i])
  }
  for pair, i := range b.IndexPairs {
    kmer1 := KmerClass{pair.Kmer1, b.Kmers[pair.Kmer1]}
    kmer2 := KmerClass{pair.Kmer2, b.Kmers[pair.Kmer2]}
    obj.AddPair(kmer1, kmer2, b.Coefficients[i])
  }
}

func (obj *KmerLrCoefficientsSet) MaxCoefficients(b *KmerLrCoefficientsSet) {
  if math.Abs(obj.Offset) < math.Abs(b.Offset) {
    obj.Offset = b.Offset
  }
  for id, i := range b.Index {
    kmer := KmerClass{id, b.Kmers[id]}
    if v := obj.Get(kmer); math.Abs(v) < math.Abs(b.Coefficients[i]) {
      obj.Set(kmer, b.Coefficients[i])
    }
  }
  for pair, i := range b.IndexPairs {
    kmer1 := KmerClass{pair.Kmer1, b.Kmers[pair.Kmer1]}
    kmer2 := KmerClass{pair.Kmer2, b.Kmers[pair.Kmer2]}
    if v := obj.GetPair(kmer1, kmer2); math.Abs(v) < math.Abs(b.Coefficients[i]) {
      obj.SetPair(kmer1, kmer2, b.Coefficients[i])
    }
  }
}

func (obj *KmerLrCoefficientsSet) MinCoefficients(b *KmerLrCoefficientsSet) {
  if math.Abs(obj.Offset) > math.Abs(b.Offset) {
    obj.Offset = b.Offset
  }
  for id, i := range obj.Index {
    kmer := KmerClass{id, obj.Kmers[id]}
    if v := b.Get(kmer); math.Abs(obj.Coefficients[i]) > math.Abs(v) {
      obj.Set(kmer, v)
    }
  }
  for pair, i := range obj.IndexPairs {
    kmer1 := KmerClass{pair.Kmer1, obj.Kmers[pair.Kmer1]}
    kmer2 := KmerClass{pair.Kmer2, obj.Kmers[pair.Kmer2]}
    if v := b.GetPair(kmer1, kmer2); math.Abs(obj.Coefficients[i]) > math.Abs(v) {
      obj.SetPair(kmer1, kmer2, v)
    }
  }
}

func (obj *KmerLrCoefficientsSet) AsKmerLr(alphabet KmerLrAlphabet) *KmerLr {
  alphabet.Kmers = obj.Kmers.AsList()
  p := alphabet.Kmers.Len()
  n := alphabet.Kmers.Len()+1
  if len(obj.IndexPairs) > 0 {
    n = (alphabet.Kmers.Len()+1)*alphabet.Kmers.Len()/2 + 1
  }
  v := NullDenseBareRealVector(n)
  for k, kmer := range alphabet.Kmers {
    v[k+1] = ConstReal(obj.Get(kmer))
  }
  if len(obj.IndexPairs) > 0 {
    for k1, kmer1 := range alphabet.Kmers {
      for k2 := k1+1; k2 < alphabet.Kmers.Len(); k2++ {
        kmer2 := alphabet.Kmers[k2]
        v[CoeffIndex(p).Ind2Sub(k1, k2)] = ConstReal(obj.GetPair(kmer1, kmer2))
      }
    }
  }
  return NewKmerLr(v, alphabet).Sparsify(nil)
}
