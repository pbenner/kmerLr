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
import   "sort"

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
  if obj.Theta.Dim() == len(obj.Kmers)+1 {
    // no co-occurrences
    for i := 0; i < len(obj.Kmers); i++ {
      if obj.Theta.ValueAt(i+1) != 0.0 {
        theta = append(theta, obj.Theta.ValueAt(i+1))
        kmers = append(kmers, obj.Kmers[i])
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
  }
  alphabet := obj.KmerLrAlphabet.Clone()
  alphabet.Kmers = kmers
  return NewKmerLr(NewDenseBareRealVector(theta), alphabet)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) GetCoefficients() *KmerLrCoefficients {
  r := NewKmerLrCoefficients()
  r.Offset = obj.Theta.ValueAt(0)
  for i := 1; i < obj.Theta.Dim(); i++ {
    r.Set(obj.Kmers[i-1], obj.Theta.ValueAt(i))
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
  c.Sort()
  *obj = *c.AsKmerLr(obj.KmerLrAlphabet.Clone())
  return nil
}

func (obj *KmerLr) Max(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MaxCoefficients(classifiers[i].GetCoefficients())
  }
  c.Sort()
  *obj = *c.AsKmerLr(obj.KmerLrAlphabet.Clone())
  return nil
}

func (obj *KmerLr) Min(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MinCoefficients(classifiers[i].GetCoefficients())
  }
  c.Sort()
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

type KmerLrCoefficients struct {
  Coefficients []float64
  Offset         float64
  Kmers          KmerClassList
  Index          map[KmerClassId]int
}

func NewKmerLrCoefficients() *KmerLrCoefficients {
  r := KmerLrCoefficients{}
  r.Index = make(map[KmerClassId]int)
  return &r
}

func (obj *KmerLrCoefficients) Add(kmer KmerClass, value float64) {
  obj.Set(kmer, obj.Get(kmer) + value)
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
      obj.Index[kmer.KmerClassId] = len(obj.Kmers)
      obj.Coefficients = append(obj.Coefficients, value)
      obj.Kmers        = append(obj.Kmers       , kmer)
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

func (obj *KmerLrCoefficients) AddCoefficients(b *KmerLrCoefficients) {
  obj.Offset += b.Offset
  for i, kmer := range b.Kmers {
    obj.Add(kmer, b.Coefficients[i])
  }
}

func (obj *KmerLrCoefficients) MaxCoefficients(b *KmerLrCoefficients) {
  if math.Abs(obj.Offset) < math.Abs(b.Offset) {
    obj.Offset = b.Offset
  }
  for i, kmer := range b.Kmers {
    if v := obj.Get(kmer); math.Abs(v) < math.Abs(b.Coefficients[i]) {
      obj.Set(kmer, b.Coefficients[i])
    }
  }
}

func (obj *KmerLrCoefficients) MinCoefficients(b *KmerLrCoefficients) {
  if math.Abs(obj.Offset) > math.Abs(b.Offset) {
    obj.Offset = b.Offset
  }
  for i, kmer := range obj.Kmers {
    if v := b.Get(kmer); math.Abs(obj.Coefficients[i]) > math.Abs(v) {
      obj.Set(kmer, v)
    }
  }
}

func (obj *KmerLrCoefficients) Len() int {
  return len(obj.Kmers)
}

func (obj *KmerLrCoefficients) Less(i, j int) bool {
  if obj.Kmers[i].K != obj.Kmers[j].K {
    return obj.Kmers[i].K < obj.Kmers[j].K
  } else {
    return obj.Kmers[i].I < obj.Kmers[j].I
  }
}

func (obj *KmerLrCoefficients) Swap(i, j int) {
  obj.Index[obj.Kmers[i].KmerClassId] = j
  obj.Index[obj.Kmers[j].KmerClassId] = i
  obj.Kmers       [i], obj.Kmers       [j] = obj.Kmers       [j], obj.Kmers       [i]
  obj.Coefficients[i], obj.Coefficients[j] = obj.Coefficients[j], obj.Coefficients[i]
}

func (obj *KmerLrCoefficients) Sort() {
  sort.Sort(obj)
}

func (obj *KmerLrCoefficients) AsKmerLr(alphabet KmerLrAlphabet) *KmerLr {
  alphabet.Kmers = obj.Kmers
  coefficients  := NewDenseBareRealVector(append([]float64{obj.Offset}, obj.Coefficients...))
  return NewKmerLr(coefficients, alphabet).Sparsify()
}
