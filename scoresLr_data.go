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

type ScoresDataSet struct {
  Data   []ConstVector
  Labels []bool
  Index  []int
  Names  []string
}

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

func read_scores_table(config Config, r io.Reader) ([]string, [][]float64, error) {
  reader := bufio.NewReader(r)
  names  :=   []string {}
  entry  := [][]float64{}
  i_     := 1
  if config.Header {
    for ;; i_++ {
      l, err := bufioReadLine(reader)
      if err == io.EOF {
        return names, entry, nil
      }
      if err != nil {
        return names, entry, err
      }
      if len(l) == 0 {
        continue
      }
      names = strings.FieldsFunc(l, func(x rune) bool { return x == ',' })
      break
    }
  }
  for ;; i_++ {
    l, err := bufioReadLine(reader)
    if err == io.EOF {
      break
    }
    if err != nil {
      return names, entry, err
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
        return names, entry, fmt.Errorf("parsing scores failed at line `%d': %v", i_, err)
      }
      entry[len(entry)-1][i] = v
    }
  }
  return names, entry, nil
}

/* -------------------------------------------------------------------------- */

func convert_scores(config Config, scores []float64, index []int, features FeatureIndices, generate_features bool) ConstVector {
  n := 0
  i := []int    {}
  v := []float64{}
  if len(features) == 0 && generate_features {
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
        c := scores[index[i1]]
        if c != 0.0 {
          i = append(i, j+1)
          v = append(v, float64(c))
        }
      } else {
        c1 := scores[index[i1]]
        c2 := scores[index[i2]]
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
  return UnsafeSparseConstFloat64Vector(i, v, n)
}

/* -------------------------------------------------------------------------- */

func import_scores(config Config, filename string, index []int, names []string, features FeatureIndices, generate_features bool, dim int) ([]ConstVector, []int, []string, int) {
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
    if granges.Length() == 0 {
      return scores, index, names, dim
    }
    data := granges.GetMeta("counts").([][]float64)
    for _, c := range data {
      if dim == -1 {
        dim = len(c)
      }
      if len(c) != dim {
        log.Fatal("Error: data has variable number of features")
      }
      if len(index) == 0 {
        index = make([]int, dim)
        for i := 0; i < dim; i++ {
          index[i] = i
        }
      }
      scores = append(scores, convert_scores(config, c, index, features, generate_features))
    }
  } else {
    if _, err := f.Seek(0, io.SeekStart); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    }
    if names_data, data, err := read_scores_table(config, f); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    } else {
      for _, c := range data {
        if dim == -1 {
          dim = len(c)
        }
        if len(c) != dim {
          PrintStderr(config, 1, "failed\n")
          log.Fatal("Error: data has variable number of features")
        }
        if len(names_data) > 0 && len(names_data) != len(c) {
          PrintStderr(config, 1, "failed\n")
          log.Fatal("Error: number of features does not match header")
        }
        scores = append(scores, convert_scores(config, c, index, features, generate_features))
      }
      if len(index) == 0 {
        index = make([]int, dim)
        for i := 0; i < dim; i++ {
          index[i] = i
        }
        // there is no index from a previous classifier, so we can
        // take the names from the data file
        names = names_data
      } else {
        // check if names match
        if len(names) != 0 && len(names_data) != 0 {
          for i, j := range index {
            if names[i] != names_data[j] {
              PrintStderr(config, 1, "failed\n")
              log.Fatalf("feature names do not match: column %d is named `%s', expected `%s'", j, names_data[j], names[i])
            }
          }
        }
      }
    }
  }
  PrintStderr(config, 1, "done\n")
  return scores, index, names, dim
}

/* -------------------------------------------------------------------------- */

func reduce_samples_scores(config Config, fg, bg []ConstVector) ([]ConstVector, []ConstVector) {
  if config.MaxSamples == 0 || len(fg) + len(bg) <= config.MaxSamples {
    return fg, bg
  }
  n1 := len(fg)
  n2 := len(bg)
  m1 := int(float64(n1)/float64(n1+n2)*float64(config.MaxSamples))
  m2 := config.MaxSamples - m1
  if m1 <= 0 || m2 <= 0 {
    log.Fatal("cannot reduce samples")
  } else {
    PrintStderr(config, 1, "Reduced training data from (%d,%d) to (%d,%d) samples\n", n1, n2, m1, m2)
  }
  return fg[0:m1], bg[0:m2]
}

/* -------------------------------------------------------------------------- */

func compile_training_data_scores(config Config, index []int, names []string, features FeatureIndices, generate_features bool, filename_fg, filename_bg string) ScoresDataSet {
  scores_fg, index, names, dim := import_scores(config, filename_fg, index, names, features, generate_features, -1)
  scores_bg,     _,     _,   _ := import_scores(config, filename_bg, index, names, features, generate_features, dim)
  scores_fg, scores_bg   = reduce_samples_scores(config, scores_fg, scores_bg)
  // define labels (assign foreground regions a label of 1)
  labels := make([]bool, len(scores_fg)+len(scores_bg))
  for i := 0; i < len(scores_fg); i++ {
    labels[i] = true
  }
  return ScoresDataSet{append(scores_fg, scores_bg...), labels, index, names}
}

func compile_test_data_scores(config Config, index []int, names []string, features FeatureIndices, generate_features bool, filename string) ScoresDataSet {
  scores, index, names, _ := import_scores(config, filename, index, names, features, generate_features, -1)
  return ScoresDataSet{scores, nil, index, names}
}

/* -------------------------------------------------------------------------- */

func compile_data_scores(config Config, index []int, names []string, features FeatureIndices, generate_features bool, filenames ...string) ([][]ConstVector, []string) {
  dim := -1
  r := make([][]ConstVector, len(filenames))
  for i, filename := range filenames {
    r[i], index, names, dim = import_scores(config, filename, index, names, features, generate_features, dim)
  }
  return r, names
}
