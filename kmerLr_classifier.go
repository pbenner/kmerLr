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
  KmerLrFeatures
  Transform
}

/* -------------------------------------------------------------------------- */

func NewKmerLr(theta Vector, alphabet KmerLrFeatures) *KmerLr {
  if lr, err := vectorDistribution.NewLogisticRegression(theta); err != nil {
    log.Fatal(err)
    return nil
  } else {
    return &KmerLr{LogisticRegression: *lr, KmerLrFeatures: alphabet}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Clone() *KmerLr {
  r := KmerLr{}
  r.LogisticRegression = *obj.LogisticRegression.Clone()
  r.KmerLrFeatures     =  obj.KmerLrFeatures    .Clone()
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

func (obj *KmerLr) Prune(data []ConstVector) *KmerLr {
  features := FeatureIndices{}
  theta    := []float64{obj.Theta.ValueAt(0)}
  kmers    := KmerClassList{}
  tr       := Transform{}
  m        := make([]int, len(obj.Kmers))
  if !obj.Cooccurrence {
    // no co-occurrences
    for i := 0; i < len(obj.Kmers); i++ {
      if obj.Theta.ValueAt(i+1) != 0.0 {
        m[i]     = len(kmers)
        theta    = append(theta, obj.Theta.ValueAt(i+1))
        kmers    = append(kmers, obj.Kmers[i])
        features = append(features, [2]int{m[obj.Features[i][0]], m[obj.Features[i][1]]})
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
        i := []int      {  0}
        v := []ConstReal{1.0}
        for it := x.ConstIterator(); it.Ok(); it.Next() {
          if j := it.Index(); j != 0 && obj.Theta.ValueAt(j) != 0.0 {
            i = append(i, m[j-1]+1)
            v = append(v, ConstReal(it.GetConst().GetValue()))
          }
        }
        // resize slice and restrict capacity
        i = append([]int      {}, i[0:len(i)]...)
        v = append([]ConstReal{}, v[0:len(v)]...)
        data[j] = UnsafeSparseConstRealVector(i, v, len(kmers)+1)
      }
    }
  } else {
    // with co-occurrences
    nz := make([]bool, len(obj.Kmers))
    md := make([]int , obj.Theta.Dim())
    md[0] = 0
    // identify kmers with non-zero coefficients
    for i, feature := range obj.Features {
      i1 := feature[0]
      i2 := feature[1]
      if obj.Theta.ValueAt(i+1) != 0.0 {
        nz[i1] = true
        nz[i2] = true
      }
    }
    // create new kmer list
    for i := 0; i < len(obj.Kmers); i++ {
      if nz[i] {
        m[i]  = len(kmers)
        kmers = append(kmers, obj.Kmers[i])
      }
    }
    // prune parameter vector and feature list
    for i := 1; i < obj.Theta.Dim(); i++ {
      if obj.Theta.ValueAt(i) != 0.0 {
        md[i]    = len(theta)
        theta    = append(theta   , obj.Theta.ValueAt(i))
        features = append(features, [2]int{m[obj.Features[i-1][0]], m[obj.Features[i-1][1]]})
      }
    }
    // prune transforms
    if len(obj.Transform.Mu) > 0 {
      tr.Mu = append(tr.Mu, obj.Transform.Mu[0])
      for i := 1; i < obj.Theta.Dim(); i++ {
        if obj.Theta.ValueAt(i) != 0.0 {
          tr.Mu = append(tr.Mu, obj.Transform.Mu[i])
        }
      }
    }
    if len(obj.Transform.Sigma) > 0 {
      tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[0])
      for i := 1; i < obj.Theta.Dim(); i++ {
        if obj.Theta.ValueAt(i) != 0.0 {
          tr.Sigma = append(tr.Sigma, obj.Transform.Sigma[i])
        }
      }
    }
    // prune data
    if len(data) != 0 {
      for j, x := range data {
        i := []int      {}
        v := []ConstReal{}
        for it := x.ConstIterator(); it.Ok(); it.Next() {
          if it.Index() == 0 || obj.Theta.ValueAt(it.Index()) != 0.0 {
            i = append(i, md[it.Index()])
            v = append(v, ConstReal(it.GetConst().GetValue()))
          }
        }
        // resize slice and restrict capacity
        i = append([]int      {}, i[0:len(i)]...)
        v = append([]ConstReal{}, v[0:len(v)]...)
        data[j] = UnsafeSparseConstRealVector(i, v, len(theta))
      }
    }
  }
  kmerLrFeatures := obj.KmerLrFeatures.Clone()
  kmerLrFeatures.Features = features
  kmerLrFeatures.Kmers    = kmers
  r := NewKmerLr(NewDenseBareRealVector(theta), kmerLrFeatures)
  r.Transform = tr
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) GetCoefficients() *KmerLrCoefficientsSet {
  r := NewKmerLrCoefficientsSet()
  r.Offset = obj.Theta.ValueAt(0)
  for i, feature := range obj.Features {
    k1 := feature[0]
    k2 := feature[1]
    if k1 == k2 {
      r.Set(obj.Kmers[k1], obj.Theta.ValueAt(i+1))
    } else {
      r.SetPair(obj.Kmers[k1], obj.Kmers[k2], obj.Theta.ValueAt(i+1))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) ExtendCooccurrence() {
  if obj.Cooccurrence == false {
    n := len(obj.Kmers)
    features := newFeatureIndices(n, true)
    theta    := make([]float64, len(features)+1)
    for i := 0; i < n+1; i++ {
      theta[i] = obj.Theta.ValueAt(i)
    }
    obj.Cooccurrence = true
    obj.Features     = features
    obj.Theta        = NewDenseBareRealVector(theta)
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Mean(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.AddCoefficients(classifiers[i].GetCoefficients())
  }
  c.DivAll(float64(len(classifiers)))
  *obj = *c.AsKmerLr(obj.KmerLrFeatures.Clone())
  return nil
}

func (obj *KmerLr) Max(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MaxCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsKmerLr(obj.KmerLrFeatures.Clone())
  return nil
}

func (obj *KmerLr) Min(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MinCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsKmerLr(obj.KmerLrFeatures.Clone())
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
  if err := obj.KmerLrFeatures.ImportConfig(config, t); err != nil {
    return err
  } else {
    if obj.Theta.Dim() != len(obj.Features)+1 {
      return fmt.Errorf("invalid config file")
    }
  }
  return nil
}

func (obj *KmerLr) ExportConfig() ConfigDistribution {
  config := obj.KmerLrFeatures.ExportConfig()
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
  config.KmerEquivalence = classifier.KmerLrFeatures.KmerEquivalence
  config.Binarize        = classifier.Binarize
  if config.Cooccurrence > 0 && classifier.Nonzero() < config.Cooccurrence {
    PrintStderr(*config, 1, "Extending parameter vector to model co-occurrence\n")
    classifier.ExtendCooccurrence()
    config.Cooccurrence = 0
  }
  return classifier
}
