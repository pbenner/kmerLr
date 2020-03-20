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
import   "os"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func predict_scores_(config Config, filename_json, filename_in string) []float64 {
  classifier := ImportScoresLrEnsemble(config, filename_json)

  data := compile_test_data_scores(config, classifier.Index, classifier.Features, filename_in)
  classifier.Transform.Apply(config, data.Data)

  predictions := classifier.Predict(config, data.Data)

  return predictions
}

func predict_scores(config Config, filename_json, filename_in, filename_out string) {
  savePredictions(filename_out, predict_scores_(config, filename_json, filename_in))
}

/* -------------------------------------------------------------------------- */

func main_predict_scores(config Config, args []string) {
  options := getopt.New()

  optHelp   := options.BoolLong("help", 'h', "print help")

  options.SetParameters("<MODEL.json> <SEQUENCES.fa> [RESULT.table]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 2 && len(options.Args()) != 3 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_json := options.Args()[0]
  filename_in   := options.Args()[1]
  filename_out  := ""
  if len(options.Args()) == 3 {
    filename_out = options.Args()[2]
  }
  predict_scores(config, filename_json, filename_in, filename_out)
}
