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
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func saveWindowPredictions(filename string, predictions [][]float64) {
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
    for j := 0; j < len(predictions[i]); j++ {
      if j == 0 {
        fmt.Fprintf(writer, "%15e", predictions[i])
      } else {
        fmt.Fprintf(writer, " %15e", predictions[i])
      }
    }
    fmt.Fprintf(writer, "\n")
  }
}

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

func predict_window(config Config, filename_json, filename_in, filename_out string, window_size, window_step int) {
  classifier := ImportKmerLr(&config, filename_json)
  sequences  := import_fasta(config, filename_in)
  job_group  := config.Pool.NewJobGroup()
  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet, classifier.Kmers...); if err != nil {
    log.Fatal(err)
  }
  predictions := make([][]float64, len(sequences))
  for i, sequence := range sequences {
    if n := len(sequence)-window_size; n > 0 {
      predictions[i] = make([]float64, n)
    }
  }
  for i, _ := range sequences {
    i        := i
    sequence := sequences[i]
    config.Pool.AddRangeJob(0, len(sequence)-window_size, job_group, func(j int, pool threadpool.ThreadPool, erf func() error) error {
      counts := scan_sequence(config, kmersCounter, []byte(sequence[i:i+window_size]))
      data   := convert_counts(config, counts, classifier.Features)
      predictions[i][j] = classifier.Predict(config, []ConstVector{data})[0]
      return nil
    })
  }
  saveWindowPredictions(filename_out, predictions)
}

/* -------------------------------------------------------------------------- */

func predict(config Config, filename_json, filename_in, filename_out string) {
  classifier := ImportKmerLr(&config, filename_json)

  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet, classifier.Kmers...); if err != nil {
    log.Fatal(err)
  }
  data := compile_test_data(config, kmersCounter, classifier.Kmers, classifier.Features, filename_in)
  classifier.Transform.Apply(config, data.Data)

  predictions := classifier.Predict(config, data.Data)

  savePredictions(filename_out, predictions)
}

/* -------------------------------------------------------------------------- */

func main_predict(config Config, args []string) {
  options := getopt.New()

  optSlidingWindow     := options.   IntLong("sliding-window",       0 ,        0, "make predictions by sliding a window along the sequence")
  optSlidingWindowStep := options.   IntLong("sliding-window-step",  0 ,        0, "step size for sliding window")
  optHelp              := options.  BoolLong("help",                'h',           "print help")

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
  if *optSlidingWindow > 0 {
    predict_window(config, filename_json, filename_in, filename_out, *optSlidingWindow, *optSlidingWindowStep)
  } else {
    predict(config, filename_json, filename_in, filename_out)
  }
}
