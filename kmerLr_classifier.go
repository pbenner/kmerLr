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

func (obj *KmerLr) Loss(data []ConstVector, c []bool, lambda float64, balance bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta.GetValues()
  lr.Lambda = lambda
  if balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  return lr.Loss(data, c, nil)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Nonzero() int {
  n  := 0
  it := obj.Theta.ConstIterator()
  if it.Ok() && it.Index() == 0 {
    it.Next()
  }
  for ; it.Ok(); it.Next() {
    if it.GetConst().GetValue() != 0.0 {
      n++
    }
  }
  return n
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Sparsify(data []ConstVector) *KmerLr {
  theta := []float64{obj.Theta.ValueAt(0)}
  kmers := KmerClassList{}
  tr    := Transform{}
  m     := make(map[int]int)
  m[0]   = 0
  if obj.Theta.Dim() == len(obj.Kmers)+1 {
    // no co-occurrences
    for i := 0; i < len(obj.Kmers); i++ {
      if obj.Theta.ValueAt(i+1) != 0.0 {
        if data != nil {
          m[i+1] = len(theta)
        }
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
    if len(data) != 0 {
      for j, x := range data {
        i := []int      {}
        v := []ConstReal{}
        for it := x.ConstIterator(); it.Ok(); it.Next() {
          if it.Index() == 0 || obj.Theta.ValueAt(it.Index()) != 0.0 {
            i = append(i, m[it.Index()])
            v = append(v, ConstReal(it.GetConst().GetValue()))
          }
        }
        data[j] = UnsafeSparseConstRealVector(i, v, len(kmers)+1)
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
        if data != nil {
          m[i+1] = len(theta)
        }
        theta = append(theta, obj.Theta.ValueAt(i))
        kmers = append(kmers, obj.Kmers[i-1])
      }
    }
    for i1 := 0; i1 < len(obj.Kmers); i1++ {
      for i2 := i1+1; i2 < len(obj.Kmers); i2++ {
        i := CoeffIndex(p).Ind2Sub(i1, i2)
        if nz[i1] && nz[i2] {
          m[i] = len(theta)
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
    if len(data) != 0 {
      n := (len(kmers)+1)*len(kmers)/2 + 1
      for j, x := range data {
        i := []int      {}
        v := []ConstReal{}
        for it := x.ConstIterator(); it.Ok(); it.Next() {
          i1, i2 := CoeffIndex(p).Sub2Ind(it.Index()-1)
          if it.Index() == 0 || (nz[i1] && nz[i2]) {
            i = append(i, m[it.Index()])
            v = append(v, ConstReal(it.GetConst().GetValue()))
          }
        }
        data[j] = UnsafeSparseConstRealVector(i, v, n)
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

func (obj *KmerLr) GetCoefficients() *KmerLrCoefficientsSet {
  r := NewKmerLrCoefficientsSet()
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

func (obj *KmerLr) ExtendCooccurrence() {
  if obj.Cooccurrence == false {
    n     := obj.Theta.Dim()-1
    theta := make([]float64, (n+1)*n/2 + 1)
    for i := 0; i < n+1; i++ {
      theta[i] = obj.Theta.ValueAt(i)
    }
    obj.Theta        = NewDenseBareRealVector(theta)
    obj.Cooccurrence = true
  }
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

func ImportKmerLr(config *Config, filename string) *KmerLr {
  classifier := new(KmerLr)
  // export model
  PrintStderr(*config, 1, "Importing distribution from `%s'... ", filename)
  if err := ImportDistribution(filename, classifier, BareRealType); err != nil {
    PrintStderr(*config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(*config, 1, "done\n")
  config.KmerEquivalence = classifier.KmerLrAlphabet.KmerEquivalence
  config.Binarize        = classifier.Binarize
  if config.Cooccurrence == 0 && !classifier.Cooccurrence {
    PrintStderr(*config, 1, "Extending parameter vector to model co-occurrence\n")
    classifier.ExtendCooccurrence()
  }
  if classifier.Cooccurrence {
    config.Cooccurrence = 0
  } else {
    config.Cooccurrence = -1
  }
  return classifier
}
