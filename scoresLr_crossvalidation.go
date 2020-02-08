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
  learnClassifiers func(i int, data_train, data_all []ConstVector, c []bool) []*ScoresLr,
   testClassifiers func(i int, data []ConstVector, classifier []*ScoresLr) [][]float64) []CVResult {
  groups := getCvGroups(len(data), config.KFoldCV, config.Seed)

  r_predictions := make([][][]float64, config.KFoldCV)
  r_labels      := make(  [][]bool,    config.KFoldCV)

  config.PoolCV.RangeJob(0, config.KFoldCV, func(i int, pool threadpool.ThreadPool, erf func() error) error {
    data_test, labels_test, data_train, labels_train := filterCvGroup(data, labels, groups, i)

    classifiers := learnClassifiers(i, data_train, data_test, labels_train)

    r_predictions[i] = testClassifiers(i, data_test, classifiers)
    r_labels     [i] = labels_test
    return nil
  })
  // join results
  result := make([]CVResult, len(config.LambdaAuto))
  for i := 0; i < config.KFoldCV; i++ {
    for j := 0; j < len(config.LambdaAuto); j++ {
      result[j].Predictions = append(result[j].Predictions, r_predictions[i][j]...)
      result[j].Labels      = append(result[j].Labels     , r_labels     [i]   ...)
    }
  }
  return result
}
