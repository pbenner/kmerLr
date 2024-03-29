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

func ImportKmerLrEnsemble(config Config, filename string) *KmerLrEnsemble {
  classifier := new(KmerLrEnsemble)
  // export model
  PrintStderr(config, 1, "Importing distribution from `%s'... ", filename)
  if err := ImportDistribution(filename, classifier, Float64Type); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  return classifier
}

/* -------------------------------------------------------------------------- */

func SaveCrossvalidation(config Config, filename string, cvr CVResult) {
  PrintStderr(config, 1, "Exporting cross-validation results to `%s'... ", filename)
  if err := saveCrossvalidation(filename, cvr); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func SaveCrossvalidationLoss(config Config, filename string, cvr CVResult) {
  PrintStderr(config, 1, "Exporting cross-validation loss to `%s'... ", filename)
  if err := saveCrossvalidationLoss(filename, cvr); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func SaveTrace(config Config, filename string, trace Trace) {
  PrintStderr(config, 1, "Exporting trace to `%s'... ", filename)
  if err := trace.Export(filename); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func SaveKmerPath(config Config, filename string, path KmerRegularizationPath) {
  PrintStderr(config, 1, "Exporting regularization path to `%s'... ", filename)
  if err := path.Export(filename); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

func SaveScoresPath(config Config, filename string, path ScoresRegularizationPath) {
  PrintStderr(config, 1, "Exporting regularization path to `%s'... ", filename)
  if err := path.Export(filename); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}

/* -------------------------------------------------------------------------- */

func SaveModel(config Config, filename string, classifier ConfigurableDistribution) {
  PrintStderr(config, 1, "Exporting model to `%s'... ", filename)
  if err := ExportDistribution(filename, classifier); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
}
