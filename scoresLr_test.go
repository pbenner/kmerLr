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

/* -------------------------------------------------------------------------- */

func TestScores1(test *testing.T) {
  config := Config{}
  config.Lambda  = 4.647556e+00
  config.Seed    = 1
  config.Verbose = 0

  main_learn_scores(config, []string{"learn", "--lambda-auto=2", "scoresLr_test_fg.table", "scoresLr_test_bg.table", "scoresLr_test"})

  classifier := ImportScoresLrEnsemble(config, "scoresLr_test_2.json").GetComponent(0)

  if len(classifier.Theta) != 3 {
    test.Error("test failed"); return
  }
  if math.Abs(classifier.Theta[0] - 0.842178566751775) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[1] - -0.05466291047449) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[2] - -0.03026279836545) > 1e-5 {
    test.Error("test failed")
  }
  if len(classifier.Features) != 2 {
    test.Error("test failed"); return
  }
  if f := classifier.Features[0]; classifier.Index[f[0]] != 1 || classifier.Index[f[1]] != 1 {
    test.Error("test failed")
  }
  if f := classifier.Features[1]; classifier.Index[f[0]] != 6 || classifier.Index[f[1]] != 6 {
    test.Error("test failed")
  }
  if v := loss_scores_(config, "scoresLr_test_2.json", "scoresLr_test_fg.table", "scoresLr_test_bg.table")[0]; math.Abs(v - 0.813659729805629) > 1e-4 {
    test.Error("test failed")
  }
  os.Remove("scoresLr_test_2.json")
}

func TestScores2(test *testing.T) {
  config := Config{}
  config.Verbose = 0
  config.Seed    = 1

  main_learn_scores(config, []string{"learn", "--lambda-auto=1", "--co-occurrence", "scoresLr_test_co_fg.table", "scoresLr_test_co_bg.table", "scoresLr_test_co"})

  classifier := ImportScoresLrEnsemble(config, "scoresLr_test_co_1.json").GetComponent(0)

  if len(classifier.Theta) != 2 {
    test.Error("test failed"); return
  }
  if math.Abs(classifier.Theta[0] - -3.6698336905701286e-06) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[1] -  0.000247759511599005) > 1e-5 {
    test.Error("test failed")
  }
  if len(classifier.Features) != 1 {
    test.Error("test failed"); return
  }
  if f := classifier.Features[0]; classifier.Index[f[0]] != 1 || classifier.Index[f[1]] != 2 {
    test.Error("test failed")
  }
  if v := loss_scores_(config, "scoresLr_test_co_1.json", "scoresLr_test_co_fg.table", "scoresLr_test_co_bg.table")[0]; math.Abs(v - 0.6699931965725273) > 1e-4 {
    test.Error("test failed")
  }
  w := []float64{
    -0.6662065007234194, -0.6477490377012717,
    -0.6539634473226117, -0.6063887787494636,
    -0.5995933447812485, -0.4461679042826637,
    -0.6290620668667339, -0.6750691863398406 }
  if v := predict_scores_(config, "scoresLr_test_co_1.json", "scoresLr_test_co_fg.table"); len(v) != len(w) {
    test.Error("test failed"); return
  } else {
    for i := 0; i < len(w); i++ {
      if math.Abs(v[i] - w[i]) > 1e-4 {
        test.Error("test failed")
      }
    }
  }
  w = []float64{
    -0.664436851779278, -0.6884446417698201,
    -0.648046764432194, -0.6861810564115558,
    -0.679086111385735, -0.6738126996571452,
    -0.688649950605459, -0.5832453150753549 }
  if v := predict_scores_(config, "scoresLr_test_co_1.json", "scoresLr_test_co_bg.table"); len(v) != len(w) {
    test.Error("test failed"); return
  } else {
    for i := 0; i < len(w); i++ {
      if math.Abs(v[i] - w[i]) > 1e-4 {
        test.Error("test failed")
      }
    }
  }
  os.Remove("scoresLr_test_co_1.json")
}

func TestScores3(test *testing.T) {
  config := Config{}
  config.Verbose = 0
  config.Seed    = 1

  filename_fg  := "scoresLr_test_co_fg.table"
  filename_bg  := "scoresLr_test_co_bg.table"
  filename_out := "scoresLr_test_co_3.json"

  main_learn_scores(config, []string{"learn", "--lambda-auto=3", "--co-occurrence", "--co-preselection=5", filename_fg, filename_bg, "scoresLr_test_co"})

  classifier := ImportScoresLrEnsemble(config, filename_out)

  data           := compile_training_data_scores(config, []int{}, []string{}, FeatureIndices{}, true, filename_fg, filename_bg)
  data_selection := classifier.SelectData(config, data)

  for i := 0; i < len(data.Data); i++ {
    for j, feature := range classifier.Features {
      i1 := classifier.Index[feature[0]]
      i2 := classifier.Index[feature[1]]
      if i1 == i2 {
        v1 := data.Data[i].Float64At(i1+1)
        v2 := data_selection[i].Float64At(j+1)
        if math.Abs(v1 - v2) > 1e-10 {
          test.Error("test failed")
        }
      } else {
        v1 := data.Data[i].Float64At(i1+1)*data.Data[i].Float64At(i2+1)
        v2 := data_selection[i].Float64At(j+1)
        if math.Abs(v1 - v2) > 1e-10 {
          test.Error("test failed")
        }
      }
    }
  }
  os.Remove(filename_out)
}

func TestScores4(test *testing.T) {
  config := Config{}
  config.Lambda  = 4.647556e+00
  config.Seed    = 1
  config.Verbose = 0

  // train kmerLr
  main_learn(config, []string{"learn", "--lambda-auto=3", "2", "6", "kmerLr_test_fg.fa", "kmerLr_test_bg.fa", "scoresLr_test_1"})

  // export kmers data
  main_export(config, []string{"export", "2", "6", "kmerLr_test_fg.fa,kmerLr_test_bg.fa", "scoresLr_test_2_fg.table,scoresLr_test_2_bg.table"})
  // train scoresLr
  main_learn_scores(config, []string{"learn", "--lambda-auto=3", "--header", "scoresLr_test_2_fg.table", "scoresLr_test_2_bg.table", "scoresLr_test_2"})

  classifier1 := ImportKmerLrEnsemble  (config, "scoresLr_test_1_3.json").GetComponent(0)
  classifier2 := ImportScoresLrEnsemble(config, "scoresLr_test_2_3.json").GetComponent(0)

  if len(classifier1.Theta) != len(classifier2.Theta) {
    test.Error("test failed")
    goto exit
  }
  for j := 0; j < len(classifier1.Theta); j++ {
    if math.Abs(classifier1.Theta[j] - classifier2.Theta[j]) > 1e-10 {
      test.Error("test failed")
      goto exit
    }
  }
  for j := 0; j < len(classifier1.Kmers); j++ {
    if classifier1.Kmers[j].String() != classifier2.Names[j] {
      test.Error("test failed")
      goto exit
    }
  }
exit:
  os.Remove("scoresLr_test_1_3.json")
  os.Remove("scoresLr_test_2_3.json")
  os.Remove("scoresLr_test_2_fg.table")
  os.Remove("scoresLr_test_2_bg.table")
}
