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

func convert_scores(config Config, scores []float64, features FeatureIndices) ConstVector {
  n := 0
  i := []int    {}
  v := []float64{}
  if len(features) == 0 {
    n = len(scores)+1
    i = make([]int    , n)
    v = make([]float64, n)
    i[0] = 0
    v[0] = 1.0
    for j := 0; j < len(scores); j++ {
      i[j+1] = j+1
      v[j+1] = scores[j]
    }
  } else {
    n = len(features)+1
    i = []int    {0  }
    v = []float64{1.0}
    for j, feature := range features {
      i1 := feature[0]
      i2 := feature[1]
      if i1 == i2 {
        c := scores[i1+1]
        if c != 0.0 {
          i = append(i, j+1)
          v = append(v, float64(c))
        }
      } else {
        c1 := scores[i1+1]
        c2 := scores[i2+1]
        if c1 != 0.0 && c2 != 0.0 {
          i = append(i, j+1)
          v = append(v, float64(c1*c2))
        }
      }
    }
  }
  // resize slice and restrict capacity
  i = append([]int    {}, i[0:len(i)]...)
  v = append([]float64{}, v[0:len(v)]...)
  return UnsafeSparseConstRealVector(i, v, n)
}

/* -------------------------------------------------------------------------- */

func import_scores(config Config, filename string, features FeatureIndices) []ConstVector {
  granges := GRanges{}
  PrintStderr(config, 1, "Reading pwm scores from `%s'... ", filename)  
  if err := granges.ImportTable(filename, []string{"counts"}, []string{"[][]float64"}); err != nil {
    PrintStderr(config, 1, "failed\n")
    panic(err)
  }
  PrintStderr(config, 1, "done\n")
  counts := []ConstVector{}
  if granges.Length() == 0 {
    return counts
  }
  for _, c := range granges.GetMeta("counts").([][]float64) {
    counts = append(counts, convert_scores(config, c, features))
  }
  return counts
}

/* -------------------------------------------------------------------------- */

func compile_training_data_scores(config Config, features FeatureIndices, filename_fg, filename_bg string) ([]ConstVector, []bool) {
  scores_fg := import_scores(config, filename_fg, features)
  scores_bg := import_scores(config, filename_bg, features)
  // define labels (assign foreground regions a label of 1)
  labels := make([]bool, len(scores_fg)+len(scores_bg))
  for i := 0; i < len(scores_fg); i++ {
    labels[i] = true
  }
  return append(scores_fg, scores_bg...), labels
}

func compile_test_data_scores(config Config, features FeatureIndices, filename string) []ConstVector {
  return import_scores(config, filename, features)
}
