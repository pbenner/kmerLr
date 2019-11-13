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
import   "math/rand"
import   "os"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func shuffleInts(vals []int, seed int64) {
  r := rand.New(rand.NewSource(seed))
  for len(vals) > 0 {
    n := len(vals)
    randIndex := r.Intn(n)
    vals[n-1], vals[randIndex] = vals[randIndex], vals[n-1]
    vals = vals[:n-1]
  }
}

/* -------------------------------------------------------------------------- */

func getCvGroups(n, fold int, seed int64) []int {
  groups := make([]int, n)
  for i := 0; i < n; i += fold {
    for j := 0; j < fold && i+j < n; j++ {
      groups[i+j] = j
    }
  }
  shuffleInts(groups, seed)
  return groups
}

func filterCvGroup(data []ConstVector, labels []bool, groups []int, i int) ([]ConstVector, []bool, []ConstVector, []bool) {
  r_test         := []ConstVector{}
  r_test_labels  := []bool{}
  r_train        := []ConstVector{}
  r_train_labels := []bool{}
  for j := 0; j < len(data); j++ {
    if groups[j] == i {
      r_test         = append(r_test        , data[j])
      r_test_labels  = append(r_test_labels , labels[j])
    } else {
      r_train        = append(r_train       , data[j])
      r_train_labels = append(r_train_labels, labels[j])
    }
  }
  return r_test, r_test_labels, r_train, r_train_labels
}

/* -------------------------------------------------------------------------- */

func saveCrossvalidation(filename string, predictions []float64, labels []bool) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  fmt.Fprintf(w, "%15s\t%6s\n", "prediction", "labels")
  for i := 0; i < len(predictions); i++ {
    if labels[i] {
      fmt.Fprintf(w, "%15e\t%6d\n", predictions[i], 1)
    } else {
      fmt.Fprintf(w, "%15e\t%6d\n", predictions[i], 0)
    }
  }
  return nil
}

/* -------------------------------------------------------------------------- */

func crossvalidation(config Config, data []ConstVector, labels []bool,
  learnClassifier func(i int, data_train, data_all []ConstVector, c []bool) *KmerLr,
   testClassifier func(i int, data []ConstVector, classifier *KmerLr) []float64) ([]float64, []bool) {
  groups := getCvGroups(len(data), config.KFoldCV, config.Seed)

  r_predictions := make([][]float64, config.KFoldCV)
  r_labels      := make([][]bool,    config.KFoldCV)

  config.PoolCV.RangeJob(0, config.KFoldCV, func(i int, pool threadpool.ThreadPool, erf func() error) error {
    data_test, labels_test, data_train, labels_train := filterCvGroup(data, labels, groups, i)

    classifier := learnClassifier(i, data_train, data_test, labels_train)

    r_predictions[i] = testClassifier(i, data_test, classifier)
    r_labels     [i] = labels_test
    return nil
  })
  // join results
  predictions := []float64{}
  labels       = []bool   {}
  for i := 0; i < config.KFoldCV; i++ {
    predictions = append(predictions, r_predictions[i]...)
    labels      = append(labels     , r_labels     [i]...)
  }
  return predictions, labels
}
