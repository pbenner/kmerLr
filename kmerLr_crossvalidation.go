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
import   "log"
import   "math/rand"
import   "os"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type CVResult struct {
  Predictions []float64
  Labels      []bool
  LambdaAuto  []int
}

/* -------------------------------------------------------------------------- */

func shuffleInts(vals1, vals2 []int, seed int64) {
  r := rand.New(rand.NewSource(seed))
  for len(vals1) > 0 {
    n := len(vals1)
    randIndex := r.Intn(n)
    vals1[n-1], vals1[randIndex] = vals1[randIndex], vals1[n-1]
    vals1 = vals1[:n-1]
    vals2[n-1], vals2[randIndex] = vals2[randIndex], vals2[n-1]
    vals2 = vals2[:n-1]
  }
}

/* -------------------------------------------------------------------------- */

func getCvGroups(n, fold int, val_ratio float64, seed int64) ([]int, []int) {
  if n < fold {
    log.Fatalf("not enough training samples (%d) for %d-fold cross-validation", n, fold)
  }
  if val_ratio > 0.0 && n < 2*fold {
    log.Fatalf("not enough training samples for creating validation set with %d-fold cross-validation", fold)
  }
  groups     := make([]int, n)
  validation := make([]int, n)
  if fold <= 1 {
    // treat special case with no cross-validation, i.e. since the group id
    // marks the test set, we have to set it to -1 so that we do not get
    // empty training sets
    for i := 0; i < n; i++ {
      groups[i] = -1
      if float64(i) < val_ratio*float64(n) && i+fold < n {
        validation[i] = 1
      } else {
        validation[i] = 0
      }
    }
  } else {
    for i := 0; i < n; i += fold {
      for j := 0; j < fold && i+j < n; j++ {
        groups[i+j] = j
        if float64(i) < val_ratio*float64(n) && i+fold < n {
          validation[i+j] = 1
        } else {
          validation[i+j] = 0
        }
      }
    }
  }
  shuffleInts(groups, validation, seed)

  return groups, validation
}

func filterCvGroup(data KmerDataSet, groups []int, validation []int, i int) (KmerDataSet, KmerDataSet, KmerDataSet) {
  r_train        := []ConstVector{}
  r_train_labels := []bool{}
  r_val          := []ConstVector{}
  r_val_labels   := []bool{}
  r_test         := []ConstVector{}
  r_test_labels  := []bool{}
  for j := 0; j < len(data.Data); j++ {
    if groups[j] == i {
      r_test        = append(r_test        , data.Data  [j])
      r_test_labels = append(r_test_labels , data.Labels[j])
    } else {
      if len(validation) > 0 && validation[j] == 1 {
        r_val        = append(r_val       , data.Data  [j])
        r_val_labels = append(r_val_labels, data.Labels[j])
      } else {
        r_train        = append(r_train       , data.Data  [j])
        r_train_labels = append(r_train_labels, data.Labels[j])
      }
    }
  }
  data_train := KmerDataSet{r_train, r_train_labels, data.Kmers}
  data_val   := KmerDataSet{r_val  , r_val_labels  , data.Kmers}
  data_test  := KmerDataSet{r_test , r_test_labels , data.Kmers}
  return data_train, data_val, data_test
}

/* -------------------------------------------------------------------------- */

func saveCrossvalidation(filename string, cvr CVResult) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  fmt.Fprintf(w, "%15s\t%6s\n", "prediction", "labels")
  for i := 0; i < len(cvr.Predictions); i++ {
    if cvr.Labels[i] {
      fmt.Fprintf(w, "%15e\t%6d\n", cvr.Predictions[i], 1)
    } else {
      fmt.Fprintf(w, "%15e\t%6d\n", cvr.Predictions[i], 0)
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func crossvalidation(config Config, data KmerDataSet,
  learnAndTestClassifiers func(i int, data_train, data_val, data_test KmerDataSet) [][]float64) []CVResult {
  groups, validation := getCvGroups(len(data.Data), config.KFoldCV, config.ValidationSize, config.Seed)

  r_predictions := make([][][]float64, config.KFoldCV)
  r_labels      := make(  [][]bool,    config.KFoldCV)

  config.PoolCV.RangeJob(0, config.KFoldCV, func(i int, pool threadpool.ThreadPool, erf func() error) error {
    config := config; config.PoolCV = pool
    i_     := i

    if config.KFoldCV <= 1 {
      i_ = -1
    }

    data_train, data_val, data_test := filterCvGroup(data, groups, validation, i)

    r_predictions[i] = learnAndTestClassifiers(i_, data_train, data_val, data_test)
    r_labels     [i] = data_test.Labels
    return nil
  })
  // join results
  result := []CVResult{}
  for i := 0; i < config.KFoldCV; i++ {
    if len(result) == 0 {
      result = make([]CVResult, len(r_predictions[i]))
    }
    if len(result) != len(r_predictions[i]) {
      panic("internal error")
    }
    for j := 0; j < len(r_predictions[i]); j++ {
      result[j].Predictions = append(result[j].Predictions, r_predictions[i][j]...)
      result[j].Labels      = append(result[j].Labels     , r_labels     [i]   ...)
    }
  }
  return result
}
