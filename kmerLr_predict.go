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

  fmt.Fprintf(w, "%15s\n", "prediction")
  for i := 0; i < len(predictions); i++ {
    fmt.Fprintf(w, "%15e\n", predictions[i])
  }
}

/* -------------------------------------------------------------------------- */

func predict_unlabeled(config Config, data []ConstVector, classifier VectorPdf) []float64 {
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

func predict_labeled(config Config, data []ConstVector, classifier VectorPdf) []float64 {
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

func predict(config Config, filename_json, filename_in, filename_out string) {
  classifier := ImportKmerLr(config, filename_json)

  // copy config from classifier
  config.KmerEquivalence = classifier.KmerLrAlphabet.KmerEquivalence
  config.Binarize        = classifier.Binarize

  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data, _ := compile_test_data(config, kmersCounter, classifier.Kmers, config.Binarize, filename_in)

  predictions := predict_unlabeled(config, data, classifier)

  savePredictions(filename_out, predictions)
}

/* -------------------------------------------------------------------------- */

func main_predict(config Config, args []string) {
  options := getopt.New()

  optHelp       := options.   BoolLong("help", 'h', "print help")

  options.SetParameters("<MODEL.json> <SEQUENCES.fa> <RESULT.table>")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 3 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_json := options.Args()[0]
  filename_in   := options.Args()[1]
  filename_out  := options.Args()[2]

  predict(config, filename_json, filename_in, filename_out)
}
