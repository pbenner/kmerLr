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
import   "math/rand"

import . "github.com/pbenner/autodiff"

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

func filterCvGroup(data []ConstVector, groups []int, i int) ([]ConstVector, []ConstVector) {
  r_test  := []ConstVector{}
  r_train := []ConstVector{}
  for j := 0; j < len(data); j++ {
    if groups[j] == i {
      r_test  = append(r_test , data[j])
    } else {
      r_train = append(r_train, data[j])
    }
  }
  return r_test, r_train
}

func getLabels(data []ConstVector) []int {
  r := make([]int, len(data))
  for i := 0; i < len(data); i++ {
    r[i] = int(data[i].ValueAt(data[i].Dim()-1))
  }
  return r
}
