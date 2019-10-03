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
import   "io"
import   "log"
import   "os"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func savePredictions(filename string, predictions []float64) {
  var writer io.Writer
  if filename == "" {
    writer = os.Stdout
  } else {
    f, err := os.Create(filename)
    if err != nil {
      panic(err)
    }
    defer f.Close()

    w := bufio.NewWriter(f)
    defer w.Flush()

    writer = w
  }
  fmt.Fprintf(writer, "%15s\n", "prediction")
  for i := 0; i < len(predictions); i++ {
    fmt.Fprintf(writer, "%15e\n", predictions[i])
  }
}

/* -------------------------------------------------------------------------- */

func predict_data(config Config, data []ConstVector, classifier VectorPdf) []float64 {
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

/* -------------------------------------------------------------------------- */

func predict(config Config, filename_json, filename_in, filename_out string) {
  classifier := ImportKmerLr(&config, filename_json)

  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data, _ := compile_test_data(config, kmersCounter, classifier.Kmers, classifier.Cooccurrence, filename_in)
  data     = classifier.TransformApply(config, data)

  predictions := predict_data(config, data, classifier)

  savePredictions(filename_out, predictions)
}

/* -------------------------------------------------------------------------- */

func main_predict(config Config, args []string) {
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
  predict(config, filename_json, filename_in, filename_out)
}
