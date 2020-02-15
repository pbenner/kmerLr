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

  classifier := ImportScoresLr(config, "scoresLr_test_2.json")

  if len(classifier.Theta) != 3 {
    test.Error("test failed"); return
  }
  if math.Abs(classifier.Theta[0] - -0.01017467785504194) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[1] - -0.5786522365231856) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[2] - -0.01642987554158162) > 1e-5 {
    test.Error("test failed")
  }
  if len(classifier.Features) != 2 {
    test.Error("test failed"); return
  }
  if f := classifier.Features[0]; f[0] != 1 || f[1] != 1 {
    test.Error("test failed")
  }
  if f := classifier.Features[1]; f[0] != 6 || f[1] != 6 {
    test.Error("test failed")
  }
  if v := loss_scores_(config, "scoresLr_test_2.json", "scoresLr_test_fg.table", "scoresLr_test_bg.table"); math.Abs(v - 10.460285) > 1e-4 {
    test.Error("test failed")
  }
  os.Remove("scoresLr_test_2.json")
}

func TestScores2(test *testing.T) {
  config := Config{}
  config.Lambda  = 5.385329e+00
  config.Verbose = 0
  config.Seed    = 1

  main_learn_scores(config, []string{"learn", "--lambda-auto=1", "--co-occurrence", "scoresLr_test_co_fg.table", "scoresLr_test_co_bg.table", "scoresLr_test_co"})

  classifier := ImportScoresLr(config, "scoresLr_test_co_1.json")

  if len(classifier.Theta) != 2 {
    test.Error("test failed"); return
  }
  if math.Abs(classifier.Theta[0] - -0.0014497332511020612) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[1] - 0.3055074741822805) > 1e-5 {
    test.Error("test failed")
  }
  if len(classifier.Features) != 1 {
    test.Error("test failed"); return
  }
  if f := classifier.Features[0]; f[0] != 1 || f[1] != 6 {
    test.Error("test failed")
  }
  if v := loss_scores_(config, "scoresLr_test_co_1.json", "scoresLr_test_co_fg.table", "scoresLr_test_co_bg.table"); math.Abs(v - 10.908436) > 1e-4 {
    test.Error("test failed")
  }
  w := []float64{
    -5.025654e-01, -6.547077e-01,
    -5.850161e-01, -5.619555e-01,
    -6.507984e-01, -3.987767e-01,
    -6.489538e-01, -6.473589e-01 }
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
    -8.531373e-01, -8.220360e-01,
    -8.446463e-01, -8.324747e-01,
    -8.784024e-01, -8.430350e-01,
    -9.011005e-01, -6.618126e-01 }
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