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
import   "math"
import   "os"
import   "testing"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func TestKmers1(test *testing.T) {
  config := Config{}
  config.Verbose  = 0

  kmersCounter, err := NewKmerCounter(4, 8, false, false, true, nil, GappedNucleotideAlphabet{}); if err != nil {
    test.Error(err)
  } else {
    data1 := compile_training_data(config, kmersCounter, nil, nil, false, "kmerLr_test.fa", "kmerLr_test.fa")

    features := newFeatureIndices(len(data1.Kmers), false)
    features  = append(features, [2]int{4671 , 4672 }) // gntanc|gntanc     = 3, gntcaa|ttganc     = 0
    features  = append(features, [2]int{5068 , 5486 }) // aaagaaa|tttcttt   = 1, aagannt|anntctt   = 7
    features  = append(features, [2]int{19270, 57071}) // aacgcgna|tncgcgtt = 1, tgaatgca|tgcattca = 1
    features  = append(features, [2]int{4671 , 5486 }) // gntanc|gntanc     = 3, aagannt|anntctt   = 7

    data2 := compile_training_data(config, kmersCounter, data1.Kmers, features, false, "kmerLr_test.fa", "kmerLr_test.fa")

    for i, _ := range features[0:len(features)-4] {
      if data1.Data[0].ValueAt(i+1) != data2.Data[0].ValueAt(i+1) {
        test.Error("test failed")
      }
      if data1.Data[1].ValueAt(i+1) != data2.Data[1].ValueAt(i+1) {
        test.Error("test failed")
      }
    }
    if data2.Data[0].ValueAt(len(features)-3) != 0 {
      test.Error("test failed")
    }
    if data2.Data[0].ValueAt(len(features)-2) != 7 {
      test.Error("test failed")
    }
    if data2.Data[0].ValueAt(len(features)-1) != 1 {
      test.Error("test failed")
    }
    if data2.Data[0].ValueAt(len(features)-0) != 21 {
      test.Error("test failed")
    }
  }
}

func TestKmers2(test *testing.T) {
  config := Config{}
  config.Verbose  = 0

  kmersCounter, err := NewKmerCounter(4, 8, false, false, true, nil, GappedNucleotideAlphabet{}); if err != nil {
    test.Error(err)
  } else {
    data1 := compile_training_data(config, kmersCounter, nil, nil, false, "kmerLr_test.fa", "kmerLr_test.fa")

    features := newFeatureIndices(len(data1.Kmers), false)

    kmersCounter, _ = NewKmerCounter(4, 8, false, false, true, nil, GappedNucleotideAlphabet{}, data1.Kmers...)
    data2 := compile_training_data(config, kmersCounter, nil, nil, false, "kmerLr_test.fa", "kmerLr_test.fa")

    if data1.Data[0].Dim() != data2.Data[0].Dim() {
      test.Error("test failed")
    } else {
      for i, _ := range features {
        if data1.Data[0].ValueAt(i+1) != data2.Data[0].ValueAt(i+1) {
          test.Error("test failed")
        }
        if data1.Data[1].ValueAt(i+1) != data2.Data[1].ValueAt(i+1) {
          test.Error("test failed")
        }
      }
    }
  }
}

func TestKmers3(test *testing.T) {
  config := Config{}
  config.Verbose        = 0
  config.MaxIterations  = 100
  config.StepSizeFactor = 1.0

  classifier := &KmerLr{}
  classifier.M        = 8
  classifier.N        = 8
  classifier.Revcomp  = true
  classifier.Alphabet = NucleotideAlphabet{}

  counter := classifier.GetKmerCounter()
  data    := compile_training_data(config, counter, nil, nil, false, "kmerLr_test_fg.fa", "kmerLr_test_bg.fa")

  estimator := NewKmerLrEstimator(config, classifier, 0)
  estimator.Estimate(config, data, TransformFull{})

  for _, x := range data.Data {
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetValue() == 0.0 {
        test.Error("test failed")
      }
    }
  }
}

func TestKmers4(test *testing.T) {
  config := Config{}
  config.Verbose        = 0
  config.StepSizeFactor = 1.0
  config.LambdaAuto     = []int{8}
  config.EvalLoss       = true
  config.EpsilonLoss    = 1e-6

  classifier := &KmerLr{}
  classifier.M        = 8
  classifier.N        = 8
  classifier.Revcomp  = true
  classifier.Alphabet = NucleotideAlphabet{}

  counter := classifier.GetKmerCounter()
  data    := compile_training_data(config, counter, nil, nil, false, "kmerLr_test_fg.fa", "kmerLr_test_bg.fa")

  estimator := NewKmerLrEstimator(config, classifier, 0)
  estimator.Estimate(config, data, TransformFull{})

  for _, x := range data.Data {
    for it := x.ConstIterator(); it.Ok(); it.Next() {
      if it.GetValue() == 0.0 {
        test.Error("test failed")
      }
    }
  }
}

func TestKmers5(test *testing.T) {
  config := Config{}
  config.Lambda  = 4.460029e+00
  config.Seed    = 1
  config.Verbose = 0

  main_learn(config, []string{"learn", "--lambda-auto=2", "--revcomp", "8", "8", "kmerLr_test_fg.fa", "kmerLr_test_bg.fa", "kmerLr_test"})

  classifier := ImportKmerLrEnsemble(config, "kmerLr_test_2.json").GetComponent(0)

  if len(classifier.Theta) != 3 {
    test.Error("test failed"); return
  }
  if math.Abs(classifier.Theta[0] - -0.002399112897182792) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[1] - -0.03289389340005168) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[2] - 0.03552066615422328) > 1e-5 {
    test.Error("test failed")
  }
  if len(classifier.Features) != 2 {
    test.Error("test failed"); return
  }
  if f := classifier.Features[0]; f[0] != 0 || f[1] != 0 {
    test.Error("test failed")
  }
  if f := classifier.Features[1]; f[0] != 1 || f[1] != 1 {
    test.Error("test failed")
  }
  if v := loss_(config, "kmerLr_test_2.json", "kmerLr_test_fg.fa", "kmerLr_test_bg.fa")[0]; math.Abs(v - 15.243974) > 1e-4 {
    test.Error("test failed")
  }
  os.Remove("kmerLr_test_2.json")
}

func TestKmers6(test *testing.T) {
  config := Config{}
  config.Lambda  = 4.987469e+00
  config.Seed    = 1
  config.Verbose = 0

  main_learn(config, []string{"learn", "--lambda-auto=2", "--binarize", "--revcomp", "--co-occurrence", "8", "8", "kmerLr_test_co_fg.fa", "kmerLr_test_co_bg.fa", "kmerLr_test_co"})

  classifier := ImportKmerLrEnsemble(config, "kmerLr_test_co_2.json").GetComponent(0)

  theta := []float64{
    -0.0067583002814767340,
     0.0063305676173333325,
     0.0063305676173333325 }
  features := [][]int{
    []int{0,1}, []int{0,2} }

  if len(classifier.Theta) != len(theta) {
    test.Error("test failed"); return
  }
  for i := 0; i < len(theta); i++ {
    if math.Abs(classifier.Theta[i] - theta[i]) > 1e-5 {
      test.Error("test failed")
    }
  }
  if len(classifier.Features) != len(features) {
    test.Error("test failed"); return
  }
  for i := 0; i < len(features); i++ {
    if f := classifier.Features[i]; f[0] != features[i][0] || f[1] != features[i][1] {
      test.Error("test failed")
    }
  }
  if v := loss_(config, "kmerLr_test_co_2.json", "kmerLr_test_co_fg.fa", "kmerLr_test_co_bg.fa")[0]; math.Abs(v - 13.862886) > 1e-4 {
    test.Error("test failed")
  }
  w := []float64{
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01,
    -6.902001e-01 }
  if v := predict_(config, "kmerLr_test_co_2.json", "kmerLr_test_co_fg.fa"); len(v) != len(w) {
    test.Error("test failed"); return
  } else {
    for i := 0; i < len(w); i++ {
      if math.Abs(v[i] - w[i]) > 1e-4 {
        test.Error("test failed")
      }
    }
  }
  w = []float64{
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01,
    -6.965320e-01 }
  if v := predict_(config, "kmerLr_test_co_2.json", "kmerLr_test_co_bg.fa"); len(v) != len(w) {
    test.Error("test failed"); return
  } else {
    for i := 0; i < len(w); i++ {
      if math.Abs(v[i] - w[i]) > 1e-4 {
        test.Error("test failed")
      }
    }
  }
  os.Remove("kmerLr_test_co_2.json")
}
