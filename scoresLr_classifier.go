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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

type ScoresLr struct {
  ScoresLrFeatures
  Theta     []float64
  Transform   Transform
}

/* -------------------------------------------------------------------------- */

func NewScoresLr(theta []float64, features ScoresLrFeatures) *ScoresLr {
  return &ScoresLr{Theta: theta, ScoresLrFeatures: features}
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Clone() *ScoresLr {
  r := ScoresLr{}
  r.Theta = make([]float64, len(obj.Theta))
  for i := 0; i < len(obj.Theta); i++ {
    r.Theta[i] = obj.Theta[i]
  }
  r.ScoresLrFeatures   =  obj.ScoresLrFeatures  .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) Predict(config Config, data []ConstVector) []float64 {
  lr := logisticRegression{}
  lr.Theta  = obj   .Theta
  lr.Lambda = config.Lambda
  lr.Pool   = config.Pool
  r := make([]float64, len(data))
  for i, _ := range data {
    r[i] = lr.LogPdf(data[i].(SparseConstRealVector))
  }
  return r
}

func (obj *ScoresLr) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Theta  = obj   .Theta
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

func (obj *ScoresLr) Nonzero() int {
  n  := 0
  for i := 1; i < len(obj.Theta); i++ {
    if obj.Theta[i] != 0.0 {
      n++
    }
  }
  return n
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLr) GetCoefficients() *ScoresLrCoefficientsSet {
  r := NewScoresLrCoefficientsSet()
  r.Offset = obj.Theta[0]
  for i, feature := range obj.Features {
    k1 := feature[0]
    k2 := feature[1]
    if k1 == k2 {
      r.Set(k1, obj.Theta[i+1])
    } else {
      r.SetPair(k1, k2, obj.Theta[i+1])
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
  if config.Name != "scoresLr" {
    return fmt.Errorf("wrong classifier type")
  }
  if len(config.Distributions) != 2 {
    return fmt.Errorf("invalid config file")
  }
  lr := vectorDistribution.LogisticRegression{}
  if err := lr.ImportConfig(config.Distributions[0], t); err != nil {
    return err
  } else {
    obj.Theta = lr.Theta.GetValues()
  }
  if err := obj.Transform.ImportConfig(config.Distributions[1], t); err != nil {
    return err
  }
  if err := obj.ScoresLrFeatures.ImportConfig(config, t); err != nil {
    return err
  } else {
    if len(obj.Theta) != len(obj.Features)+1 {
      return fmt.Errorf("invalid config file")
    }
  }
  return nil
}

func (obj *ScoresLr) ExportConfig() ConfigDistribution {
  if lr, err := vectorDistribution.NewLogisticRegression(NewDenseBareRealVector(obj.Theta)); err != nil {
    panic("internal error")
  } else {
    config := obj.ScoresLrFeatures.ExportConfig()
    config.Name          = "scoresLr"
    config.Distributions = []ConfigDistribution{
      lr           .ExportConfig(),
      obj.Transform.ExportConfig() }
    return config
  }
}
