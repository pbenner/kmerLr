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

func (obj *KmerLr) Predict(config Config, data []ConstVector) []float64 {
  r := make([]float64, len(data))
  t := BareReal(0.0)
  for i, _ := range data {
    if err := obj.LogPdf(&t, data[i]); err != nil {
      log.Fatal(err)
    }
    r[i] = t.GetValue()
  }
  return r
}

func (obj *KmerLr) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta.GetValues()
  lr.Lambda = config.Lambda
  if config.Balance {
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
    if it.GetValue() != 0.0 {
      n++
    }
  }
  return n
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
  config.Cooccurrence    = classifier.Cooccurrence || config.Cooccurrence
  config.KmerEquivalence = classifier.KmerLrFeatures.KmerEquivalence
  config.Binarize        = classifier.Binarize
  return classifier
}
