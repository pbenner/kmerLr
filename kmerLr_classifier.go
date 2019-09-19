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
import   "log"
import   "math"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLr struct {
  vectorDistribution.LogisticRegression
  KmerLrAlphabet
  Transform
}

/* -------------------------------------------------------------------------- */

func NewKmerLr(theta Vector, alphabet KmerLrAlphabet) *KmerLr {
  if lr, err := vectorDistribution.NewLogisticRegression(theta); err != nil {
    log.Fatal(err)
    return nil
  } else {
    return &KmerLr{LogisticRegression: *lr, KmerLrAlphabet: alphabet}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Clone() *KmerLr {
  r := KmerLr{}
  r.LogisticRegression = *obj.LogisticRegression.Clone()
  r.KmerLrAlphabet     =  obj.KmerLrAlphabet    .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Loss(data []ConstVector, lambda float64, balance bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta.GetValues()
  lr.Lambda = lambda
  if balance {
    lr.ClassWeights = compute_class_weights(data)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  return lr.Loss(data, nil)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Sparsify() *KmerLr {
  theta := []float64{obj.Theta.ValueAt(0)}
  kmers := KmerClassList{}
  tr    := Transform{}
  if obj.Theta.Dim() == len(obj.Kmers)+1 {
    // no co-occurrences
    for i := 0; i < len(obj.Kmers); i++ {
      if obj.Theta.ValueAt(i+1) != 0.0 {
        theta = append(theta, obj.Theta.ValueAt(i+1))
        kmers = append(kmers, obj.Kmers[i])
      }
    }
    if len(obj.Transform.Mu) > 0 {
      tr.Mu    = append(tr.Mu   , obj.Transform.Mu   [0])
      tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[0])
      for i := 0; i < len(obj.Kmers); i++ {
        if obj.Theta.ValueAt(i+1) != 0.0 {
          tr.Mu    = append(tr.Mu   , obj.Transform.Mu   [i+1])
          tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[i+1])
        }
      }
    }
  } else {
    // with co-occurrences
    p  := len(obj.Kmers)
    nz := make([]bool, len(obj.Kmers))
    for i := 0; i < len(obj.Kmers); i++ {
      if obj.Theta.ValueAt(i+1) != 0.0 {
        nz[i] = true
      }
    }
    for i1 := 0; i1 < len(obj.Kmers); i1++ {
      for i2 := i1+1; i2 < len(obj.Kmers); i2++ {
        i := CoeffIndex(p).Ind2Sub(i1, i2)
        if obj.Theta.ValueAt(i) != 0.0 {
          nz[i1] = true
          nz[i2] = true
        }
      }
    }
    for i := 1; i <= len(obj.Kmers); i++ {
      if nz[i-1] {
        theta = append(theta, obj.Theta.ValueAt(i))
        kmers = append(kmers, obj.Kmers[i-1])
      }
    }
    for i1 := 0; i1 < len(obj.Kmers); i1++ {
      for i2 := i1+1; i2 < len(obj.Kmers); i2++ {
        i := CoeffIndex(p).Ind2Sub(i1, i2)
        if nz[i1] && nz[i2] {
          theta = append(theta, obj.Theta.ValueAt(i))
        }
      }
    }
    if len(obj.Transform.Mu) > 0 {
      tr.Mu    = append(tr.Mu   , obj.Transform.Mu   [0])
      tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[0])
      for i := 0; i < len(obj.Kmers); i++ {
        if nz[i] {
          tr.Mu    = append(tr.Mu   , obj.Transform.Mu   [i+1])
          tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[i+1])
        }
      }
      for i1 := 0; i1 < len(obj.Kmers); i1++ {
        for i2 := i1+1; i2 < len(obj.Kmers); i2++ {
          i := CoeffIndex(p).Ind2Sub(i1, i2)
          if nz[i1] && nz[i2] {
            tr.Mu    = append(tr.Mu   , obj.Transform.Mu   [i])
            tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[i])
          }
        }
      }
    }
  }
  alphabet := obj.KmerLrAlphabet.Clone()
  alphabet.Kmers = kmers
  r := NewKmerLr(NewDenseBareRealVector(theta), alphabet)
  r.Transform = tr
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) GetCoefficients() *KmerLrCoefficients {
  r := NewKmerLrCoefficients()
  p := obj.Kmers.Len()
  r.Offset = obj.Theta.ValueAt(0)
  for i, kmer := range obj.Kmers {
    r.Set(kmer, obj.Theta.ValueAt(i+1))
  }
  if obj.Theta.Dim() > p+1 {
    for k1, kmer1 := range obj.Kmers {
      for k2 := k1+1; k2 < p; k2++ {
        kmer2 := obj.Kmers[k2]
        r.SetPair(kmer1, kmer2, obj.Theta.ValueAt(CoeffIndex(p).Ind2Sub(k1, k2)))
      }
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Mean(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.AddCoefficients(classifiers[i].GetCoefficients())
  }
  c.DivAll(float64(len(classifiers)))
  *obj = *c.AsKmerLr(obj.KmerLrAlphabet.Clone())
  return nil
}

func (obj *KmerLr) Max(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MaxCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsKmerLr(obj.KmerLrAlphabet.Clone())
  return nil
}

func (obj *KmerLr) Min(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MinCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsKmerLr(obj.KmerLrAlphabet.Clone())
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) ImportConfig(config ConfigDistribution, t ScalarType) error {
  if len(config.Distributions) != 2 {
    return fmt.Errorf("invalid config file")
  }
  if err := obj.LogisticRegression.ImportConfig(config.Distributions[0], t); err != nil {
    return err
  }
  if err := obj.Transform.ImportConfig(config.Distributions[1], t); err != nil {
    return err
  }
  return obj.KmerLrAlphabet.ImportConfig(config, t)
}

func (obj *KmerLr) ExportConfig() ConfigDistribution {
  config := obj.KmerLrAlphabet.ExportConfig()
  config.Name          = "kmerLr"
  config.Distributions = []ConfigDistribution{
    obj.LogisticRegression.ExportConfig(),
    obj.Transform         .ExportConfig() }

  return config
}

/* -------------------------------------------------------------------------- */

func ImportKmerLr(config Config, filename string) *KmerLr {
  classifier := new(KmerLr)
  // export model
  PrintStderr(config, 1, "Importing distribution from `%s'... ", filename)
  if err := ImportDistribution(filename, classifier, BareRealType); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  if config.Cooccurrence && !classifier.Cooccurrence {
    PrintStderr(config, 1, "Extending parameter vector to model co-occurrence\n")
    n     := classifier.Theta.Dim()-1
    theta := make([]float64, (n+1)*n/2 + 1)
    for i := 0; i < n+1; i++ {
      theta[i] = classifier.Theta.ValueAt(i)
    }
    classifier.Theta        = NewDenseBareRealVector(theta)
    classifier.Cooccurrence = config.Cooccurrence
  }
  return classifier
}

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

type KmerLrCoefficients struct {
  Offset         float64
  Coefficients []float64
  Kmers          KmerClassSet
  Index          map[KmerClassId    ]int
  IndexPairs     map[KmerClassIdPair]int
}

func NewKmerLrCoefficients() *KmerLrCoefficients {
  r := KmerLrCoefficients{}
  r.Kmers      = make(KmerClassSet)
  r.Index      = make(map[KmerClassId]int)
  r.IndexPairs = make(map[KmerClassIdPair]int)
  return &r
}

func (obj *KmerLrCoefficients) Add(kmer KmerClass, value float64) {
  obj.Set(kmer, obj.Get(kmer) + value)
}

func (obj *KmerLrCoefficients) AddPair(kmer1, kmer2 KmerClass, value float64) {
  obj.SetPair(kmer1, kmer2, obj.GetPair(kmer1, kmer2) + value)
}

func (obj *KmerLrCoefficients) DivAll(value float64) {
  obj.Offset /= value
  for i, _ := range obj.Coefficients {
    obj.Coefficients[i] /= value
  }
}

func (obj *KmerLrCoefficients) Set(kmer KmerClass, value float64) {
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

func (obj *KmerLrCoefficients) SetPair(kmer1, kmer2 KmerClass, value float64) {
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

func (obj *KmerLrCoefficients) Get(kmer KmerClass) float64 {
  if i, ok := obj.Index[kmer.KmerClassId]; ok {
    return obj.Coefficients[i]
  } else {
    return 0.0
  }
}

func (obj *KmerLrCoefficients) GetPair(kmer1, kmer2 KmerClass) float64 {
  pair := NewKmerClassIdPair(kmer1, kmer2)
  if i, ok := obj.IndexPairs[pair]; ok {
    return obj.Coefficients[i]
  } else {
    return 0.0
  }
}

func (obj *KmerLrCoefficients) AddCoefficients(b *KmerLrCoefficients) {
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

func (obj *KmerLrCoefficients) MaxCoefficients(b *KmerLrCoefficients) {
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

func (obj *KmerLrCoefficients) MinCoefficients(b *KmerLrCoefficients) {
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

func (obj *KmerLrCoefficients) AsKmerLr(alphabet KmerLrAlphabet) *KmerLr {
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
  return NewKmerLr(v, alphabet).Sparsify()
}
