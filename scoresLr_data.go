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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func import_scores(config Config, filename string) []ConstVector {
  granges := GRanges{}
  PrintStderr(config, 1, "Reading pwm scores from `%s'... ", filename)  
  if err := granges.ImportTable(filename, []string{"counts"}, []string{"[][]float64"}); err != nil {
    PrintStderr(config, 1, "failed\n")
    panic(err)
  }
  PrintStderr(config, 1, "done\n")
  counts := []ConstVector{}
  for _, c := range granges.GetMeta("counts").([][]float64) {
    counts = append(counts, AsSparseConstRealVector(NewVector(BareRealType, c)))
  }
  return counts
}

/* -------------------------------------------------------------------------- */

func compile_training_data_scores(config Config, filename_fg, filename_bg string) ([]ConstVector, []bool) {
  scores_fg := import_scores(config, filename_fg)
  scores_bg := import_scores(config, filename_bg)
  // define labels (assign foreground regions a label of 1)
  labels := make([]bool, len(scores_fg)+len(scores_bg))
  for i := 0; i < len(scores_fg); i++ {
    labels[i] = true
  }
  return append(scores_fg, scores_bg...), labels
}

func compile_test_data_scores(config Config, filename string) []ConstVector {
  return import_scores(config, filename)
}
