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

type ScoresLr struct {
  vectorDistribution.LogisticRegression
  ScoresLrFeatures
  Transform Transform
}

/* -------------------------------------------------------------------------- */

func NewScoresLr(theta Vector, alphabet ScoresLrFeatures) *ScoresLr {
  if lr, err := vectorDistribution.NewLogisticRegression(theta); err != nil {
    log.Fatal(err)
    return nil
  } else {
    return &ScoresLr{LogisticRegression: *lr, ScoresLrFeatures: alphabet}
  }
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Clone() *ScoresLr {
  r := ScoresLr{}
  r.LogisticRegression = *obj.LogisticRegression.Clone()
  r.ScoresLrFeatures   =  obj.ScoresLrFeatures  .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Predict(config Config, data []ConstVector) []float64 {
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

func (obj *ScoresLr) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj.Theta.GetValues()
  lr.Lambda = config.Lambda
  if config.Balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  return lr.Loss(data, c)
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Nonzero() int {
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

func (obj *ScoresLr) GetCoefficients() *ScoresLrCoefficientsSet {
  r := NewScoresLrCoefficientsSet()
  r.Offset = obj.Theta.ValueAt(0)
  for i, feature := range obj.Features {
    k1 := feature[0]
    k2 := feature[1]
    if k1 == k2 {
      r.Set(k1, obj.Theta.ValueAt(i+1))
    } else {
      r.SetPair(k1, k2, obj.Theta.ValueAt(i+1))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) JoinTransforms(classifiers []*ScoresLr) error {
  if len(classifiers) == 0 || classifiers[0].Transform.Nil() {
    return nil
  }
  obj.Transform = NewTransform(len(obj.Features)+1)
  for _, classifier := range classifiers {
    if err := obj.Transform.InsertScores(classifier.Transform, obj.Features, classifier.Features); err != nil {
      return err
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Mean(classifiers []*ScoresLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.AddCoefficients(classifiers[i].GetCoefficients())
  }
  c.DivAll(float64(len(classifiers)))
  *obj = *c.AsScoresLr(obj.ScoresLrFeatures)
  return obj.JoinTransforms(classifiers)
}

func (obj *ScoresLr) Max(classifiers []*ScoresLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MaxCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsScoresLr(obj.ScoresLrFeatures)
  return obj.JoinTransforms(classifiers)
}

func (obj *ScoresLr) Min(classifiers []*ScoresLr) error {
  c := classifiers[0].GetCoefficients()
  for i := 1; i < len(classifiers); i++ {
    c.MinCoefficients(classifiers[i].GetCoefficients())
  }
  *obj = *c.AsScoresLr(obj.ScoresLrFeatures)
  return obj.JoinTransforms(classifiers)
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) ImportConfig(config ConfigDistribution, t ScalarType) error {
  if len(config.Distributions) != 2 {
    return fmt.Errorf("invalid config file")
  }
  if err := obj.LogisticRegression.ImportConfig(config.Distributions[0], t); err != nil {
    return err
  }
  if err := obj.Transform.ImportConfig(config.Distributions[1], t); err != nil {
    return err
  }
  if err := obj.ScoresLrFeatures.ImportConfig(config, t); err != nil {
    return err
  } else {
    if obj.Theta.Dim() != len(obj.Features)+1 {
      return fmt.Errorf("invalid config file")
    }
  }
  return nil
}

func (obj *ScoresLr) ExportConfig() ConfigDistribution {
  config := obj.ScoresLrFeatures.ExportConfig()
  config.Name          = "scoresLr"
  config.Distributions = []ConfigDistribution{
    obj.LogisticRegression.ExportConfig(),
    obj.Transform         .ExportConfig() }

  return config
}

/* -------------------------------------------------------------------------- */

func ImportScoresLr(config *Config, filename string) *ScoresLr {
  classifier := new(ScoresLr)
  // export model
  PrintStderr(*config, 1, "Importing distribution from `%s'... ", filename)
  if err := ImportDistribution(filename, classifier, BareRealType); err != nil {
    PrintStderr(*config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(*config, 1, "done\n")
  config.Cooccurrence    = classifier.Cooccurrence || config.Cooccurrence
  return classifier
}
