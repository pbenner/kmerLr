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
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func scoresCrossvalidation(config Config, data []ConstVector, labels []bool,
  learnClassifier func(i int, data_train, data_test []ConstVector, c []bool) *ScoresLr,
   testClassifier func(i int, data []ConstVector, classifier *ScoresLr) []float64) ([]float64, []bool) {
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
