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
import   "log"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLr struct {
  KmerLrFeatures
  Theta     []float64
  Transform   Transform
}

/* -------------------------------------------------------------------------- */

func NewKmerLr(theta []float64, alphabet KmerLrFeatures) *KmerLr {
  return &KmerLr{Theta: theta, KmerLrFeatures: alphabet}
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Clone() *KmerLr {
  r := KmerLr{}
  r.Theta = make([]float64, len(obj.Theta))
  for i := 0; i < len(obj.Theta); i++ {
    r.Theta[i] = obj.Theta[i]
  }
  r.KmerLrFeatures = obj.KmerLrFeatures.Clone()
  r.Transform      = obj.Transform     .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Predict(config Config, data []ConstVector) []float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.Pool
  r := make([]float64, len(data))
  for i, _ := range data {
    r[i] = lr.LogPdf(data[i].(SparseConstRealVector))
  }
  return r
}

func (obj *KmerLr) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.Pool
  if config.Balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  return lr.Loss(data, c)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Nonzero() int {
  n  := 0
  for i := 1; i < len(obj.Theta); i++ {
    if obj.Theta[i] != 0.0 {
      n++
    }
  }
  return n
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) GetCoefficients() *KmerLrCoefficientsSet {
  r := NewKmerLrCoefficientsSet()
  r.Offset = obj.Theta[0]
  for i, feature := range obj.Features {
    k1 := feature[0]
    k2 := feature[1]
    if k1 == k2 {
      r.Set(obj.Kmers[k1], obj.Theta[i+1])
    } else {
      r.SetPair(obj.Kmers[k1], obj.Kmers[k2], obj.Theta[i+1])
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) GetKmerCounter() *KmerCounter {
  if counter, err := NewKmerCounter(obj.M, obj.N, obj.Complement, obj.Reverse, obj.Revcomp, obj.MaxAmbiguous, obj.Alphabet, obj.Kmers...); err != nil {
    log.Fatal(err)
    return nil
  } else {
    return counter
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) JoinTransforms(classifiers []*KmerLr) error {
  if len(classifiers) == 0 || classifiers[0].Transform.Nil() {
    return nil
  }
  obj.Transform = NewTransform(len(obj.Features)+1)
  for _, classifier := range classifiers {
    if err := obj.Transform.Insert(classifier.Transform, obj.Features, classifier.Features, obj.Kmers, classifier.Kmers); err != nil {
      return err
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Mean(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.AddCoefficients(classifiers[i].GetCoefficients())
  }
  c.DivAll(float64(len(classifiers)))
  *obj = *c.AsKmerLr(obj.KmerLrFeatures)
  return obj.JoinTransforms(classifiers)
}

func (obj *KmerLr) Max(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MaxCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsKmerLr(obj.KmerLrFeatures)
  return obj.JoinTransforms(classifiers)
}

func (obj *KmerLr) Min(classifiers []*KmerLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MinCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsKmerLr(obj.KmerLrFeatures)
  return obj.JoinTransforms(classifiers)
}

/* -------------------------------------------------------------------------- */

type jointKmerLr struct {
  classifiers   []*KmerLrEnsemble
  counters    [][]*KmerCounter
}

func importJointKmerLr(config Config, filenames []string) jointKmerLr {
  counters    := make([][]*KmerCounter   , len(filenames))
  classifiers := make(  []*KmerLrEnsemble, len(filenames))
  for i, filename := range filenames {
    classifiers[i] = ImportKmerLrEnsemble(config, filename)
    counters   [i] = make([]*KmerCounter, config.Pool.NumberOfThreads())
    for j := 0; j < config.Pool.NumberOfThreads(); j++ {
      counters[i][j] = classifiers[i].GetKmerCounter()
    }
  }
  return jointKmerLr{classifiers, counters}
}

func (obj jointKmerLr) Predict(config Config, subseq []byte, j int) float64 {
  r := 0.0
  for i, _ := range obj.classifiers {
    counts := scan_sequence(config, obj.counters[i][j], obj.classifiers[i].Binarize, subseq)
    counts.SetKmers(obj.classifiers[i].Kmers)
    data   := convert_counts(config, counts, obj.classifiers[i].Features)
    r      += obj.classifiers[i].Predict(config, []ConstVector{data})[0]
  }
  return r
}
