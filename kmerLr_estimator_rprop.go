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
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/rprop"
import . "github.com/pbenner/autodiff/statistics"
import   "github.com/pbenner/autodiff/statistics/vectorDistribution"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type KmerLrRpropEstimator struct {
  logisticRegression
  Balance        bool
  Epsilon        float64
  MaxIterations  int
  Hook           rprop.Hook
  StepSize       float64
  StepSizeFactor float64
  Eta          []float64
  data         []ConstVector
  labels       []bool
  // list of all features
  Kmers     KmerClassList
  Transform Transform
}

/* -------------------------------------------------------------------------- */

func NewKmerLrRpropEstimator(config Config, kmers KmerClassList, trace *Trace, icv int, data []ConstVector, labels []bool, t Transform) *KmerLrRpropEstimator {
  n := kmers.Len() + 1
  if config.Cooccurrence == 0 {
    n = (kmers.Len()+1)*kmers.Len()/2 + 1
  }
  r := KmerLrRpropEstimator{}
  r.Balance        = config.Balance
  r.ClassWeights   = [2]float64{1, 1}
  r.Lambda         = config.Lambda
  r.Epsilon        = config.Epsilon
  r.MaxIterations  = config.MaxEpochs
  r.StepSize       = config.RpropStepSize
  r.StepSizeFactor = config.StepSizeFactor
  r.Eta            = config.RpropEta
  r.Kmers          = kmers
  r.Transform      = t
  r.Theta          = make([]float64, n)
  r.Hook           = NewRpropHook(config, trace, icv, data, labels, &r)
  return &r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) Clone() *KmerLrRpropEstimator {
  panic("internal error")
}

func (obj *KmerLrRpropEstimator) CloneVectorEstimator() VectorEstimator {
  panic("internal error")
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) GetParameters() Vector {
  return NewDenseBareRealVector(obj.Theta)
}

func (obj *KmerLrRpropEstimator) SetParameters(x Vector) error {
  obj.Theta = x.GetValues()
  return nil
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) Estimate(config Config, data []ConstVector, labels []bool) *KmerLr {
  if obj.Balance {
    obj.computeClassWeights(labels)
  }
  args := []interface{}{}
  args  = append(args, rprop.Epsilon{obj.Epsilon})
  args  = append(args, obj.Hook)
  if obj.MaxIterations > 0 {
    args = append(args, rprop.MaxIterations{obj.MaxIterations})
  }
  if obj.StepSize == 0.0 {
    obj.setStepSize(data)
  }
  obj.data   = data
  obj.labels = labels
  x, err := rprop.RunGradient(rprop.DenseGradientF(obj.objectiveGradient), DenseConstRealVector(obj.Theta), obj.StepSize, obj.Eta, args...)
  obj.data = nil
  if err != nil {
    log.Fatal(err)
    return nil
  }
  if lr, err := vectorDistribution.NewLogisticRegression(NewVector(BareRealType, x.GetValues())); err != nil {
    log.Fatal(err)
    return nil
  } else {
    r := KmerLr{}
    r.LogisticRegression             = *lr
    r.Transform                      = obj.   Transform
    r.Cooccurrence                   = config.Cooccurrence == 0
    r.KmerLrFeatures.Binarize        = config.Binarize
    r.KmerLrFeatures.KmerEquivalence = config.KmerEquivalence
    r.KmerLrFeatures.Kmers           = obj   .Kmers
    return r.Prune(nil)
  }
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrRpropEstimator) objectiveGradient(theta, gradient DenseConstRealVector) error {
  obj.Theta = theta
  obj.Gradient(gradient, obj.data, obj.labels, nil)
  return nil
}

func (obj *KmerLrRpropEstimator) computeClassWeights(labels []bool) {
  obj.ClassWeights = compute_class_weights(labels)
}

func (obj *KmerLrRpropEstimator) setStepSize(data []ConstVector) {
  max_squared_sum := 0.0
  for i, _ := range data {
    r  := 0.0
    it := data[i].ConstIterator()
    // skip first element
    if it.Ok() {
      it.Next()
    }
    for ; it.Ok(); it.Next() {
      // skip last element
      if it.Index() == data[i].Dim()-1 {
        break
      }
      r += it.GetValue()*it.GetValue()
    }
    if r > max_squared_sum {
      max_squared_sum = r
    }
  }
  L := 0.25*(max_squared_sum + 1.0)
  L *= math.Max(obj.ClassWeights[0], obj.ClassWeights[1])
  obj.StepSize  = 1.0/L
  obj.StepSize *= obj.StepSizeFactor
}
