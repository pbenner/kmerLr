/* Copyright (C) 2020 Philipp Benner
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
import   "regexp"
import   "sort"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

/* -------------------------------------------------------------------------- */

type ScoresLrEnsemble struct {
  ScoresLrFeatures
  Theta     [][]float64
  Transform     Transform
  Summary       string
}

/* -------------------------------------------------------------------------- */

func NewScoresLrEnsemble(summary string) *ScoresLrEnsemble {
  return &ScoresLrEnsemble{Summary: summary}
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) Clone() *ScoresLrEnsemble {
  r := ScoresLrEnsemble{}
  r.Theta = make([][]float64, len(obj.Theta))
  for i := 0; i < len(obj.Theta); i++ {
    r.Theta[i] = make([]float64, len(obj.Theta[i]))
    for j := 0; j < len(obj.Theta); j++ {
      r.Theta[i][j] = obj.Theta[i][j]
    }
  }
  r.ScoresLrFeatures = obj.ScoresLrFeatures.Clone()
  r.Transform        = obj.Transform       .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) Summarize(config Config, x []float64) float64 {
  if len(x) == 0 {
    return math.NaN()
  }
  var r float64
  switch obj.Summary {
  case "mean":
    r = 0.0
    for j := 0; j < len(x); j++ {
      r += x[j]
    }
    r = r/float64(len(x))
  case "product":
    r = 1.0
    for j := 0; j < len(x); j++ {
      r *= x[j]
    }
  case "min":
    r = x[0]
    for j := 1; j < len(x); j++ {
      if r > x[j] {
        r = x[j]
      }
    }
  case "max":
    r = x[0]
    for j := 1; j < len(x); j++ {
      if r < x[j] {
        r = x[j]
      }
    }
  case "":
    if len(x) == 1 {
      r = x[0]
    } else {
      log.Fatal("no summary given for ensemble classifier")
    }
  default:
    panic("internal error")
  }
  return r
}

func (obj *ScoresLrEnsemble) Loss(config Config, data []ConstVector, c []bool) []float64 {
  lr := logisticRegression{}
  lr.Lambda = config.Lambda
  //lr.Pool   = config.Pool
  if config.Balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  r := make([]float64, len(obj.Theta))
  for i, _ := range obj.Theta {
    lr.Theta = obj.Theta[i]
    r[i] = lr.Loss(data, c)
  }
  return r
}

func (obj *ScoresLrEnsemble) Predict(config Config, data []ConstVector) []float64 {
  lr := logisticRegression{}
  lr.Lambda = config.Lambda
  //lr.Pool   = config.Pool
  r := make([]float64, len(data))
  t := make([]float64, len(obj.Theta))
  for i, _ := range data {
    for j := 0; j < len(obj.Theta); j++ {
      lr.Theta = obj.Theta[j]
      t[j] = lr.LogPdf(data[i].(SparseConstRealVector))
    }
    r[i] = obj.Summarize(config, t)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) EnsembleSize() int {
  return len(obj.Theta)
}

func (obj *ScoresLrEnsemble) GetComponent(i int) *ScoresLr {
  r := ScoresLr{}
  if i >= len(obj.Theta) {
    r.Theta = nil
  } else {
    r.Theta = obj.Theta[i]
  }
  r.ScoresLrFeatures = obj.ScoresLrFeatures
  r.Transform        = obj.Transform
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) Mean() *ScoresLrEnsemble {
  if len(obj.Theta) <= 1 {
    return obj
  }
  t := &ScoresLr{}
  t.ScoresLrFeatures = obj.ScoresLrFeatures
  t.Transform        = obj.Transform
  t.Theta = make([]float64, len(obj.Theta[0]))
  for j := 0; j < len(obj.Theta[0]); j++ {
    for i := 0; i < len(obj.Theta); i++ {
      t.Theta[j] += obj.Theta[i][j]
    }
    t.Theta[j] /= float64(len(obj.Theta))
  }
  r := &ScoresLrEnsemble{}
  r.AddScoresLr(t)
  return r
}

func (obj *ScoresLrEnsemble) Min() *ScoresLrEnsemble {
  if len(obj.Theta) <= 1 {
    return obj
  }
  t := &ScoresLr{}
  t.ScoresLrFeatures = obj.ScoresLrFeatures
  t.Transform        = obj.Transform
  t.Theta = make([]float64, len(obj.Theta[0]))
  for j := 0; j < len(obj.Theta[0]); j++ {
    t.Theta[j] += obj.Theta[0][j]
    for i := 1; i < len(obj.Theta); i++ {
      if math.Abs(t.Theta[j]) > math.Abs(obj.Theta[i][j]) {
        t.Theta[j] = obj.Theta[i][j]
      }
    }
  }
  r := &ScoresLrEnsemble{}
  r.AddScoresLr(t)
  return r
}

func (obj *ScoresLrEnsemble) Max() *ScoresLrEnsemble {
  if len(obj.Theta) <= 1 {
    return obj
  }
  t := &ScoresLr{}
  t.ScoresLrFeatures = obj.ScoresLrFeatures
  t.Transform        = obj.Transform
  t.Theta = make([]float64, len(obj.Theta[0]))
  for j := 0; j < len(obj.Theta[0]); j++ {
    t.Theta[j] += obj.Theta[0][j]
    for i := 1; i < len(obj.Theta); i++ {
      if math.Abs(t.Theta[j]) < math.Abs(obj.Theta[i][j]) {
        t.Theta[j] = obj.Theta[i][j]
      }
    }
  }
  r := &ScoresLrEnsemble{}
  r.AddScoresLr(t)
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) AddScoresLr(classifier *ScoresLr) error {
  if len(obj.Theta) == 0 {
    obj.Cooccurrence = classifier.Cooccurrence
  }
  if obj.Cooccurrence != classifier.Cooccurrence {
    return fmt.Errorf("co-occurrence is not consistent across classifiers")
  }
  if !obj.Transform.Nil() && !obj.Transform.EqualsScores(classifier.Transform, obj.Features, classifier.Features) {
    return fmt.Errorf("data transform is not consistent across classifiers")
  }
  n  := len(obj.Theta)
  m1 := make(map[[2]int]int)
  m2 := make(map[[2]int]int)
  ki := make(map[int]int)
  z  := make(map[int]struct{})
  // map kmer pairs to feature indices (ensemble classifier)
  for i, feature := range obj.ScoresLrFeatures.Features {
    i1 := obj.ScoresLrFeatures.Index[feature[0]]
    i2 := obj.ScoresLrFeatures.Index[feature[1]]
    m1[[2]int{i1,i2}] = i
    z [i1] = struct{}{}
    z [i2] = struct{}{}
  }
  // map kmer pairs to feature indices (new ScoresLr classifier)
  for i, feature := range classifier.ScoresLrFeatures.Features {
    // filter out zero coefficients
    if classifier.Theta[i+1] != 0.0 {
      i1 := classifier.ScoresLrFeatures.Index[feature[0]]
      i2 := classifier.ScoresLrFeatures.Index[feature[1]]
      m2[[2]int{i1,i2}] = i
      z [i1] = struct{}{}
      z [i2] = struct{}{}
    }
  }
  // create index union
  index := []int{}
  for i, _ := range z {
    index = append(index, i)
  }
  sort.Ints(index)
  // create map
  for i, j := range index {
    ki[j] = i
  }
  // construct new feature set
  features := FeatureIndices{}
  for idx, _ := range m1 {
    i1 := ki[idx[0]]
    i2 := ki[idx[1]]
    features = append(features, [2]int{i1,i2})
  }
  for idx, _ := range m2 {
    if _, ok := m1[idx]; !ok {
      i1 := ki[idx[0]]
      i2 := ki[idx[1]]
      features = append(features, [2]int{i1,i2})
    }
  }
  features.Sort()
  coefficients := make([][]float64, n+1)
  // copy coefficients of ensemble classifier
  for i := 0; i < n; i++ {
    coefficients[i]    = make([]float64, len(features)+1)
    coefficients[i][0] = obj.Theta[i][0]
    for j, feature := range features {
      i1 := index[feature[0]]
      i2 := index[feature[1]]
      if k, ok := m1[[2]int{i1,i2}]; ok {
        coefficients[i][j+1] = obj.Theta[i][k+1]
      }
    }
  }
  // copy coefficients of new kmerLr classifier
  coefficients[n]    = make([]float64, len(features)+1)
  coefficients[n][0] = classifier.Theta[0]
  for j, feature := range features {
    i1 := index[feature[0]]
    i2 := index[feature[1]]
    if k, ok := m2[[2]int{i1,i2}]; ok {
      coefficients[n][j+1] = classifier.Theta[k+1]
    }
  }
  // create new transform
  transform := Transform{}
  if !classifier.Transform.Nil() {
    transform = NewTransform(len(features)+1)
    if !obj.Transform.Nil() {
      if err := transform.InsertScores(obj.Transform, features, obj.Features); err != nil {
        return err
      }
    }
    if err := transform.InsertScores(classifier.Transform, features, classifier.Features); err != nil {
      return err
    }
  }
  obj.ScoresLrFeatures.Features = features
  obj.ScoresLrFeatures.Index    = index
  obj.Theta                     = coefficients
  obj.Transform                 = transform
  return nil
}

func (obj *ScoresLrEnsemble) AddScoresLrEnsemble(classifier *ScoresLrEnsemble) error {
  if obj.Summary != "" && obj.Summary != classifier.Summary {
    return fmt.Errorf("ensemble summary is not consistent across classifiers")
  }
  if obj.Summary == "" {
    obj.Summary = classifier.Summary
  }
  for i := 0; i < classifier.EnsembleSize(); i++ {
    if err := obj.AddScoresLr(classifier.GetComponent(i)); err != nil {
      return err
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) SelectData(config Config, data_ ScoresDataSet) []ConstVector {
  data     := data_.Data
  data_dst := make([]ConstVector, len(data))
  index    := data_.Index
  imap     := make(map[int]int)
  for i, j := range index {
    imap[j] = i
  }
  for i_ := 0; i_ < len(data); i_++ {
    i := []int    {                 0 }
    v := []float64{data[i_].ValueAt(0)}
    for j, feature := range obj.Features {
      if feature[0] == feature[1] {
        i1, ok := imap[feature[0]]; if !ok {
          panic("internal error")
        }
        if value := data[i_].ValueAt(i1+1); value != 0.0 {
          i = append(i, j+1)
          v = append(v, value)
        }
      } else {
        i1, ok := imap[feature[0]]; if !ok {
          panic("internal error")
        }
        i2, ok := imap[feature[1]]; if !ok {
          panic("internal error")
        }
        if value := data[i_].ValueAt(i1+1)*data[i_].ValueAt(i2+1); value != 0.0 {
          i = append(i, j+1)
          v = append(v, value)
        }
      }
    }
    // resize slice and restrict capacity
    i = append([]int    {}, i[0:len(i)]...)
    v = append([]float64{}, v[0:len(v)]...)
    data_dst[i_] = UnsafeSparseConstRealVector(i, v, len(obj.Features)+1)
  }
  obj.Transform.Apply(config, data_dst)
  return data_dst
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrEnsemble) ImportConfig(config ConfigDistribution, t ScalarType) error {
  re := regexp.MustCompile(`scoresLr \[([a-zA-Z]+)\]`)
  if config.Name != "scoresLr" && !re.MatchString(config.Name) {
    return fmt.Errorf("wrong classifier type")
  }
  if len(config.Distributions) < 2 {
    return fmt.Errorf("invalid config file")
  }
  lr := vectorDistribution.LogisticRegression{}
  n  := len(config.Distributions)
  obj.Theta = make([][]float64, n-1)
  for j := 0; j < n-1; j++ {
    if err := lr.ImportConfig(config.Distributions[j], t); err != nil {
      return err
    } else {
      obj.Theta[j] = lr.Theta.GetValues()
    }
  }
  if err := obj.Transform.ImportConfig(config.Distributions[n-1], t); err != nil {
    return err
  }
  if err := obj.ScoresLrFeatures.ImportConfig(config, t); err != nil {
    return err
  } else {
    if len(obj.Theta[0]) != len(obj.Features)+1 {
      return fmt.Errorf("invalid config file")
    }
  }
  if config.Name == "scoresLr" {
    obj.Summary = ""
  } else {
    obj.Summary = re.FindStringSubmatch(config.Name)[1]
  }
  return nil
}

func (obj *ScoresLrEnsemble) ExportConfig() ConfigDistribution {
  distributions := []ConfigDistribution{}
  for j := 0; j < len(obj.Theta); j++ {    
    if lr, err := vectorDistribution.NewLogisticRegression(NewDenseBareRealVector(obj.Theta[j])); err != nil {
      panic("internal error")
    } else {
      distributions = append(distributions, lr.ExportConfig())
    }
  }
  distributions = append(distributions, obj.Transform.ExportConfig())
  config := obj.ScoresLrFeatures.ExportConfig()
  if obj.Summary == "" {
    config.Name = fmt.Sprintf("scoresLr")
  } else {
    config.Name = fmt.Sprintf("scoresLr [%s]", obj.Summary)
  }
  config.Distributions = distributions
  return config
}
