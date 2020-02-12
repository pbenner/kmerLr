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
import   "testing"

/* -------------------------------------------------------------------------- */

func TestScores1(test *testing.T) {
  config := Config{}
  config.Verbose = 0
  config.Lambda  = 4.647556e+00

  main_learn_scores(config, []string{"learn", "--lambda-auto=2", "scoresLr_test_fg.table", "scoresLr_test_bg.table", "scoresLr_test"})

  classifier := ImportScoresLr(config, "scoresLr_test_2.json")

  if len(classifier.Theta) != 3 {
    test.Error("test failed"); return
  }
  if math.Abs(classifier.Theta[0] - -0.012583476345881155) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[1] - -0.5800261197863501) > 1e-5 {
    test.Error("test failed")
  }
  if math.Abs(classifier.Theta[2] - -0.01890505530675889) > 1e-5 {
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
}
