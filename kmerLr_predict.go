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
import   "log"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

func predict(config Config, classifier VectorPdf, data []ConstVector) []float64 {
  r := make([]float64, len(data))
  t := BareReal(0.0)
  for i, _ := range data {
    if err := classifier.LogPdf(&t, data[i]); err != nil {
      log.Fatal(err)
    }
    r[i] = t.GetValue()
  }
  return r
}

func predict_labeled(config Config, classifier VectorPdf, data []ConstVector) []float64 {
  r := make([]float64, len(data))
  t := BareReal(0.0)
  for i, _ := range data {
    // drop label, i.e. first component
    if err := classifier.LogPdf(&t, data[i].ConstSlice(0, data[i].Dim()-1)); err != nil {
      log.Fatal(err)
    }
    r[i] = t.GetValue()
  }
  return r
}
