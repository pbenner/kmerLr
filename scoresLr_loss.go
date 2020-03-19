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
import   "os"
import   "strconv"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func loss_scores_(config Config, filename_json, filename_fg, filename_bg string) []float64 {
  classifier := ImportScoresLrEnsemble(config, filename_json)
  data       := compile_training_data_scores(config, classifier.Features, filename_fg, filename_bg)
  classifier.Transform.Apply(config, data.Data)

  return classifier.Loss(config, data.Data, data.Labels)
}

func loss_scores(config Config, filename_json, filename_fg, filename_bg, filename_out string) {
  saveLoss(filename_out, loss_scores_(config, filename_json, filename_fg, filename_bg))
}

/* -------------------------------------------------------------------------- */

func main_loss_scores(config Config, args []string) {
  options := getopt.New()

  optBalance := options.  BoolLong("balance", 0 ,        "set class weights so that the data set is balanced")
  optLambda  := options.StringLong("lambda",  0 , "0.0", "regularization strength (L1)")
  optHelp    := options.  BoolLong("help",   'h',        "print help")

  options.SetParameters("<MODEL.json> <FOREGROUND.fa> <BACKGROUND.fa> [RESULT.table]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 3 && len(options.Args()) != 4 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if v, err := strconv.ParseFloat(*optLambda, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Lambda = v
  }
  config.Balance = *optBalance

  filename_json := options.Args()[0]
  filename_fg   := options.Args()[1]
  filename_bg   := options.Args()[2]
  filename_out  := ""
  if len(options.Args()) == 4 {
    filename_out = options.Args()[3]
  }

  loss_scores(config, filename_json, filename_fg, filename_bg, filename_out)
}
