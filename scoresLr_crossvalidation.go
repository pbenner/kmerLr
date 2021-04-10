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

func scoresFilterCvGroup(data ScoresDataSet, groups, validation []int, i int) (ScoresDataSet, ScoresDataSet, ScoresDataSet) {
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
  data_train := ScoresDataSet{r_train, r_train_labels, data.Index, data.Names}
  data_val   := ScoresDataSet{r_val  , r_val_labels  , data.Index, data.Names}
  data_test  := ScoresDataSet{r_test , r_test_labels , data.Index, data.Names}
  return data_train, data_val, data_test
}

func scoresCrossvalidation(config Config, data ScoresDataSet,
  learnAndTestClassifiers func(i int, data_train, data_val, data_test ScoresDataSet) [][]float64) []CVResult {
  groups, validation := getCvGroups(len(data.Data), config.KFoldCV, config.ValidationSize, config.Seed)

  r_predictions := make([][][]float64, config.KFoldCV)
  r_labels      := make(  [][]bool,    config.KFoldCV)

  config.PoolCV.RangeJob(0, config.KFoldCV, func(i int, pool threadpool.ThreadPool, erf func() error) error {
    config := config; config.PoolCV = pool
    i_     := i

    if config.KFoldCV <= 1 {
      i_ = -1
    }

    data_train, data_val, data_test := scoresFilterCvGroup(data, groups, validation, i)

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
