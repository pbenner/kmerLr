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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLrEnsemble struct {
  KmerLrFeatures
  Theta     [][]float64
  Transform     Transform
  Summary       string
}

/* -------------------------------------------------------------------------- */

func NewKmerLrEnsemble(summary string) *KmerLrEnsemble {
  return &KmerLrEnsemble{Summary: summary}
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) Clone() *KmerLrEnsemble {
  r := KmerLrEnsemble{}
  r.Theta = make([][]float64, len(obj.Theta))
  for i := 0; i < len(obj.Theta); i++ {
    r.Theta[i] = make([]float64, len(obj.Theta[i]))
    for j := 0; j < len(obj.Theta); j++ {
      r.Theta[i][j] = obj.Theta[i][j]
    }
  }
  r.KmerLrFeatures = obj.KmerLrFeatures.Clone()
  r.Transform      = obj.Transform     .Clone()
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) Summarize(config Config, x []float64) float64 {
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

func (obj *KmerLrEnsemble) Loss(config Config, data []ConstVector, c []bool) float64 {
  lr := logisticRegression{}
  lr.Lambda = config.Lambda
  lr.Pool   = config.PoolLR
  if config.Balance {
    lr.ClassWeights = compute_class_weights(c)
  } else {
    lr.ClassWeights[0] = 1.0
    lr.ClassWeights[1] = 1.0
  }
  r := make([]float64, len(obj.Theta))
  for j, _ := range obj.Theta {
    lr.Theta = obj.Theta[j]
    r[j] = lr.Loss(data, c)
  }
  return obj.Summarize(config, r)
}

func (obj *KmerLrEnsemble) Predict(config Config, data []ConstVector) []float64 {
  lr := logisticRegression{}
  lr.Lambda = config.Lambda
  lr.Pool   = config.PoolLR
  r := make([]float64, len(data))
  t := make([]float64, len(obj.Theta))
  for i, _ := range data {
    for j, _ := range obj.Theta {
      lr.Theta = obj.Theta[j]
      t[j] = lr.LogPdf(data[i].(SparseConstFloat64Vector))
    }
    r[i] = obj.Summarize(config, t)
  }
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) GetKmerCounter() *KmerCounter {
  if counter, err := NewKmerCounter(obj.M, obj.N, obj.Complement, obj.Reverse, obj.Revcomp, obj.MaxAmbiguous, obj.Alphabet, obj.Kmers...); err != nil {
    log.Fatal(err)
    return nil
  } else {
    return counter
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) EnsembleSize() int {
  return len(obj.Theta)
}

func (obj *KmerLrEnsemble) GetComponent(i int) *KmerLr {
  r := KmerLr{}
  if i >= len(obj.Theta) {
    r.Theta        = nil
  } else {
    r.Theta        = obj.Theta[i]
  }
  r.KmerLrFeatures = obj.KmerLrFeatures
  r.Transform      = obj.Transform
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) Mean() *KmerLrEnsemble {
  if len(obj.Theta) <= 1 {
    return obj
  }
  t := &KmerLr{}
  t.KmerLrFeatures = obj.KmerLrFeatures
  t.Transform      = obj.Transform
  t.Theta          = make([]float64, len(obj.Theta[0]))
  for j := 0; j < len(obj.Theta[0]); j++ {
    for i := 0; i < len(obj.Theta); i++ {
      t.Theta[j] += obj.Theta[i][j]
    }
    t.Theta[j] /= float64(len(obj.Theta))
  }
  r := &KmerLrEnsemble{}
  r.AddKmerLr(t)
  return r
}

func (obj *KmerLrEnsemble) Min() *KmerLrEnsemble {
  if len(obj.Theta) <= 1 {
    return obj
  }
  t := &KmerLr{}
  t.KmerLrFeatures = obj.KmerLrFeatures
  t.Transform      = obj.Transform
  t.Theta          = make([]float64, len(obj.Theta[0]))
  for j := 0; j < len(obj.Theta[0]); j++ {
    t.Theta[j] += obj.Theta[0][j]
    for i := 1; i < len(obj.Theta); i++ {
      if math.Abs(t.Theta[j]) > math.Abs(obj.Theta[i][j]) {
        t.Theta[j] = obj.Theta[i][j]
      }
    }
  }
  r := &KmerLrEnsemble{}
  r.AddKmerLr(t)
  return r
}

func (obj *KmerLrEnsemble) Max() *KmerLrEnsemble {
  if len(obj.Theta) <= 1 {
    return obj
  }
  t := &KmerLr{}
  t.KmerLrFeatures = obj.KmerLrFeatures
  t.Transform      = obj.Transform
  t.Theta          = make([]float64, len(obj.Theta[0]))
  for j := 0; j < len(obj.Theta[0]); j++ {
    t.Theta[j] += obj.Theta[0][j]
    for i := 1; i < len(obj.Theta); i++ {
      if math.Abs(t.Theta[j]) < math.Abs(obj.Theta[i][j]) {
        t.Theta[j] = obj.Theta[i][j]
      }
    }
  }
  r := &KmerLrEnsemble{}
  r.AddKmerLr(t)
  return r
}

func (obj *KmerLrEnsemble) Stability() ([]float64, []float64) {
  if len(obj.Theta) == 0 {
    return nil, nil
  }
  x := make([]float64, len(obj.Theta[0])-1)
  y := make([]float64, len(obj.Theta[0])-1)
  for j := 1; j < len(obj.Theta[0]); j++ {
    for i := 0; i < len(obj.Theta); i++ {
      if obj.Theta[i][j] != 0.0 {
        x[j-1] += 1.0
        y[j-1] += math.Abs(obj.Theta[i][j])
      }
    }
    y[j-1] = y[j-1] / x[j-1]
  }
  return x, y
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) AddKmerLr(classifier *KmerLr) error {
  if len(obj.Theta) == 0 {
    obj.KmerLrEquivalence = classifier.KmerLrEquivalence
  }
  if err := obj.KmerLrEquivalence.Equals(classifier.KmerLrEquivalence); err != nil {
    return err
  }
  if !obj.Transform.Nil() && !obj.Transform.Equals(classifier.Transform, obj.Features, classifier.Features, obj.Kmers, classifier.Kmers) {
    return fmt.Errorf("data transform is not consistent across classifiers")
  }
  n  := len(obj.Theta)
  m1 := make(map[[2]KmerClassId]int)
  m2 := make(map[[2]KmerClassId]int)
  ki := make(map[   KmerClassId]int)
  z  := make(map[   KmerClassId][]string)
  // map kmer pairs to feature indices (ensemble classifier)
  for i, feature := range obj.KmerLrFeatures.Features {
    kmer1 := obj.KmerLrFeatures.Kmers[feature[0]]
    kmer2 := obj.KmerLrFeatures.Kmers[feature[1]]
    m1[[2]KmerClassId{kmer1.KmerClassId, kmer2.KmerClassId}] = i
    z[kmer1.KmerClassId] = kmer1.Elements
    z[kmer2.KmerClassId] = kmer2.Elements
  }
  // map kmer pairs to feature indices (new KmerLr classifier)
  for i, feature := range classifier.KmerLrFeatures.Features {
    // filter out zero coefficients
    if classifier.Theta[i+1] != 0.0 {
      kmer1 := classifier.KmerLrFeatures.Kmers[feature[0]]
      kmer2 := classifier.KmerLrFeatures.Kmers[feature[1]]
      m2[[2]KmerClassId{kmer1.KmerClassId, kmer2.KmerClassId}] = i
      z[kmer1.KmerClassId] = kmer1.Elements
      z[kmer2.KmerClassId] = kmer2.Elements
    }
  }
  // create union of kmers
  kmers := KmerClassList{}
  for id, elem := range z {
    kmers = append(kmers, KmerClass{id, elem})
  }
  kmers.Sort()
  // create map
  for i, kmer := range kmers {
    ki[kmer.KmerClassId] = i
  }
  // construct new feature set
  features := FeatureIndices{}
  for kmers, _ := range m1 {
    i1 := ki[kmers[0]]
    i2 := ki[kmers[1]]
    features = append(features, [2]int{i1,i2})
  }
  for kmers, _ := range m2 {
    if _, ok := m1[kmers]; !ok {
      i1 := ki[kmers[0]]
      i2 := ki[kmers[1]]
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
      kmer1 := kmers[feature[0]].KmerClassId
      kmer2 := kmers[feature[1]].KmerClassId
      if k, ok := m1[[2]KmerClassId{kmer1, kmer2}]; ok {
        coefficients[i][j+1] = obj.Theta[i][k+1]
      }
    }
  }
  // copy coefficients of new kmerLr classifier
  coefficients[n]    = make([]float64, len(features)+1)
  coefficients[n][0] = classifier.Theta[0]
  for j, feature := range features {
    kmer1 := kmers[feature[0]].KmerClassId
    kmer2 := kmers[feature[1]].KmerClassId
    if k, ok := m2[[2]KmerClassId{kmer1, kmer2}]; ok {
      coefficients[n][j+1] = classifier.Theta[k+1]
    }
  }
  // create new transform
  transform := Transform{}
  if !classifier.Transform.Nil() {
    transform = NewTransform(len(features), len(classifier.Transform.Offset) > 0, len(classifier.Transform.Scale) > 0)
    if !obj.Transform.Nil() {
      if err := transform.Insert(obj.Transform, features, obj.Features, kmers, obj.Kmers); err != nil {
        return err
      }
    }
    if err := transform.Insert(classifier.Transform, features, classifier.Features, kmers, classifier.Kmers); err != nil {
      return err
    }
  }
  obj.KmerLrFeatures.Features = features
  obj.KmerLrFeatures.Kmers    = kmers
  obj.Theta                   = coefficients
  obj.Transform               = transform
  return nil
}

func (obj *KmerLrEnsemble) AddKmerLrEnsemble(classifier *KmerLrEnsemble) error {
  if obj.Summary != "" && obj.Summary != classifier.Summary {
    return fmt.Errorf("ensemble summary is not consistent across classifiers")
  }
  if obj.Summary == "" {
    obj.Summary = classifier.Summary
  }
  for i := 0; i < classifier.EnsembleSize(); i++ {
    if err := obj.AddKmerLr(classifier.GetComponent(i)); err != nil {
      return err
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) SelectData(config Config, data KmerDataSet) []ConstVector {
  r := KmerLr{obj.KmerLrFeatures, nil, obj.Transform}
  return r.SelectData(config, data)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrEnsemble) ImportConfig(config ConfigDistribution, t ScalarType) error {
  re := regexp.MustCompile(`kmerLr \[([a-zA-Z]+)\]`)
  if config.Name != "kmerLr" && !re.MatchString(config.Name) {
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
      obj.Theta[j] = AsDenseFloat64Vector(lr.Theta)
    }
  }
  if err := obj.Transform.ImportConfig(config.Distributions[n-1], t); err != nil {
    return err
  }
  if err := obj.KmerLrFeatures.ImportConfig(config, t); err != nil {
    return err
  } else {
    if len(obj.Theta[0]) != len(obj.Features)+1 {
      return fmt.Errorf("invalid config file")
    }
  }
  if config.Name == "kmerLr" {
    obj.Summary = ""
  } else {
    obj.Summary = re.FindStringSubmatch(config.Name)[1]
  }
  return nil
}

func (obj *KmerLrEnsemble) ExportConfig() ConfigDistribution {
  distributions := []ConfigDistribution{}
  for j := 0; j < len(obj.Theta); j++ {    
    if lr, err := vectorDistribution.NewLogisticRegression(NewDenseFloat64Vector(obj.Theta[j])); err != nil {
      panic("internal error")
    } else {
      distributions = append(distributions, lr.ExportConfig())
    }
  }
  distributions = append(distributions, obj.Transform.ExportConfig())
  config := obj.KmerLrFeatures.ExportConfig()
  if obj.Summary == "" {
    config.Name = fmt.Sprintf("kmerLr")
  } else {
    config.Name = fmt.Sprintf("kmerLr [%s]", obj.Summary)
  }
  config.Distributions = distributions
  return config
}
