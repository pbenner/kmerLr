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
import   "log"
import   "os"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

func PrintStderr(config Config, level int, format string, args ...interface{}) {
  if config.Verbose >= level {
    fmt.Fprintf(os.Stderr, format, args...)
  }
}

/* -------------------------------------------------------------------------- */

func SaveCrossvalidation(config Config, filename string, predictions []float64, labels []int) {
  PrintStderr(config, 1, "Exporting cross-validation results to `%s'... ", filename)
  if err := saveCrossvalidation(filename, predictions, labels); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func SaveTrace(config Config, filename string, trace *Trace) {
  PrintStderr(config, 1, "Exporting trace to `%s'... ", filename)
  if err := trace.Export(filename); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func SaveModel(config Config, filename string, classifier VectorPdf, z []float64) {
  if z != nil {
    PrintStderr(config, 1, "Scaling model parameters... ")
    theta := classifier.GetParameters()
    for i, zi := range z {
      theta.At(i).Mul(theta.ConstAt(i), ConstReal(zi))
    }
    classifier.SetParameters(theta)
    PrintStderr(config, 1, "done\n")
  }
  PrintStderr(config, 1, "Exporting model to `%s'... ", filename)
  if err := ExportDistribution(filename, classifier); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}
