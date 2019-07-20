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

/* -------------------------------------------------------------------------- */

type KmerLr struct {
  vectorDistribution.LogisticRegression
  KmerLrAlphabet
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
  r.KmerLrAlphabet     =  obj.KmerLrAlphabet
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) Mean(classifiers []*KmerLr) error {
  obj.Theta.Reset()
  for _, classifier := range classifiers {
    obj.Theta.VaddV(obj.Theta, classifier.Theta)
  }
  n := ConstReal(float64(len(classifiers)))
  for i := 0; i < obj.Theta.Dim(); i++ {
    obj.Theta.At(i).Div(obj.Theta.ConstAt(i), n)
  }
  return nil
}

func (obj *KmerLr) Max(classifiers []*KmerLr) error {
  for i := 0; i < obj.Theta.Dim(); i++ {
    obj.Theta.At(i).SetValue(0.0)
  }
  maxAbs := func(a Scalar, b ConstScalar) {
    if math.Abs(a.GetValue()) < math.Abs(b.GetValue()) {
      a.Set(b)
    }
  }
  for i := 0; i < obj.Theta.Dim(); i++ {
    for _, classifier := range classifiers {
      maxAbs(obj.Theta.At(i), classifier.Theta.ConstAt(i))
    }
  }
  return nil
}

func (obj *KmerLr) Min(classifiers []*KmerLr) error {
  for i := 0; i < obj.Theta.Dim(); i++ {
    obj.Theta.At(i).SetValue(math.Inf(1))
  }
  minAbs := func(a Scalar, b ConstScalar) {
    if math.Abs(a.GetValue()) > math.Abs(b.GetValue()) {
      a.Set(b)
    }
  }
  for i := 0; i < obj.Theta.Dim(); i++ {
    for _, classifier := range classifiers {
      minAbs(obj.Theta.At(i), classifier.Theta.ConstAt(i))
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLr) ImportConfig(config ConfigDistribution, t ScalarType) error {
  if len(config.Distributions) != 1 {
    return fmt.Errorf("invalid config file")
  }
  if err := obj.LogisticRegression.ImportConfig(config.Distributions[0], t); err != nil {
    return err
  }
  return obj.KmerLrAlphabet.ImportConfig(config, t)
}

func (obj *KmerLr) ExportConfig() ConfigDistribution {
  config := obj.KmerLrAlphabet.ExportConfig()
  config.Name          = "kmerLr"
  config.Distributions = []ConfigDistribution{
    obj.LogisticRegression.ExportConfig() }

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
  return classifier
}
