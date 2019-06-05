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
import   "log"
import   "os"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func savePredictions(filename string, predictions []float64) {
  f, err := os.Create(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  fmt.Fprintf(w, "prediction\n")
  for i := 0; i < len(predictions); i++ {
    fmt.Fprintf(w, "%15f\n", predictions[i])
  }
}

/* -------------------------------------------------------------------------- */

func predict_unlabeled(config Config, classifier VectorPdf, data []ConstVector) []float64 {
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

/* -------------------------------------------------------------------------- */

func predict(config Config, filename_in, filename_out string) {
  classifier := new(KmerLr)
  // export model
  PrintStderr(config, 1, "Importing distribution from `%s'... ", filename_in)
  if err := ImportDistribution(filename_in, classifier, BareRealType); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")

  config.M, config.N  = classifier.M, classifier.N
  // copy config from classifier
  config.Binarize     = classifier.Binarize
  config.Complement   = classifier.Complement
  config.Reverse      = classifier.Reverse
  config.Revcomp      = classifier.Revcomp
  config.MaxAmbiguous = classifier.MaxAmbiguous
  config.Alphabet     = classifier.Alphabet

  kmersCounter, err := NewKmersCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data := compile_test_data(config, kmersCounter, config.Binarize, filename_in)

  predictions := predict_unlabeled(config, classifier, data)

  savePredictions(filename_out, predictions)
}

/* -------------------------------------------------------------------------- */

func main_predict(config Config, args []string) {
  options := getopt.New()

  optVerbose    := options.CounterLong("verbose",   'v', "verbose level [-v or -vv]")
  optHelp       := options.   BoolLong("help",      'h', "print help")

  options.SetParameters("<SEQUENCES.fa> <RESULT.table>")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optVerbose != 0 {
    config.Verbose = *optVerbose
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 2 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_in  := options.Args()[0]
  filename_out := options.Args()[1]

  predict(config, filename_in, filename_out)
}
