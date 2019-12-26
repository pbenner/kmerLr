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
import   "bufio"
import   "io"
import   "log"
import   "os"
import   "strconv"
import   "strings"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func bufioReadLine(reader *bufio.Reader) (string, error) {
  l, err := reader.ReadString('\n')
  if err != nil {
    // ignore EOF errors if some bytes were read
    if len(l) > 0 && err == io.EOF {
      return l, nil
    }
    return l, err
  }
  // remove newline character
  return l[0:len(l)-1], err
}

func read_scores_table(r io.Reader) ([][]float64, error) {
  reader := bufio.NewReader(r)
  entry  := [][]float64{}
  for i_ := 1;; i_++ {
    l, err := bufioReadLine(reader)
    if err == io.EOF {
      break
    }
    if err != nil {
      return entry, err
    }
    if len(l) == 0 {
      continue
    }
    data := strings.FieldsFunc(l, func(x rune) bool { return x == ',' })
    entry = append(entry, make([]float64, len(data)))
    // loop over count vector
    for i := 0; i < len(data); i++ {
      v, err := strconv.ParseFloat(strings.TrimSpace(data[i]), 64)
      if err != nil {
        return entry, fmt.Errorf("parsing scores failed at line `%d': %v", i_, err)
      }
      entry[len(entry)-1][i] = v
    }
  }
  return entry, nil
}

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
        c := scores[i1]
        if c != 0.0 {
          i = append(i, j+1)
          v = append(v, float64(c))
        }
      } else {
        c1 := scores[i1]
        c2 := scores[i2]
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
  f, err := os.Open(filename)
  if err != nil {
    log.Fatal(err)
  }
  defer f.Close()

  scores  := []ConstVector{}
  granges := GRanges{}
  PrintStderr(config, 1, "Reading scores from `%s'... ", filename)
  if err := granges.ReadTable(f, []string{"counts"}, []string{"[][]float64"}); err == nil {
    // scores are in GRanges format
    PrintStderr(config, 1, "done\n")
    if granges.Length() == 0 {
      return scores
    }
    data := granges.GetMeta("counts").([][]float64)
    for _, c := range data {
      if len(c) != len(data[0]) {
        log.Fatal("Error: data has variable number of features")
      }
      scores = append(scores, convert_scores(config, c, features))
    }
  } else {
    if _, err := f.Seek(0, io.SeekStart); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    }
    if data, err := read_scores_table(f); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    } else {
      PrintStderr(config, 1, "done\n")
      for _, c := range data {
        if len(c) != len(data[0]) {
          log.Fatal("Error: data has variable number of features")
        }
        scores = append(scores, convert_scores(config, c, features))
      }
    }
  }
  return scores
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
