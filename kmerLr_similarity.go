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
import   "math"
import   "io"
import   "os"

import   "github.com/pborman/getopt"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func write_similarity_matrix(config Config, similarities [][]float64, filenameOut string) {
  var writer io.Writer

  if filenameOut == "" {
    writer = os.Stdout
  } else {
    f, err := os.Create(filenameOut)
    if err != nil {
      log.Fatal(err)
    }
    buffer := bufio.NewWriter(f)
    writer  = buffer
    defer f.Close()
    defer buffer.Flush()
  }
  for _, x := range similarities {
    for _, xi := range x {
      fmt.Fprintf(writer, "%8.4f ", xi)
    }
    fmt.Fprintf(writer, "\n")
  }
}

/* -------------------------------------------------------------------------- */

func compute_similarity(config Config, theta ConstVector, x1, x2 ConstVector) float64 {
  r0 := 0.0
  r1 := 0.0
  r2 := 0.0
  for it := theta.ConstIterator(); it.Ok(); it.Next() {
    i  := it.Index()
    vt := it.GetValue()
    v1 := x1.ValueAt(i)
    v2 := x2.ValueAt(i)
    r0 += v1*vt*v2
    r1 += v1*vt*v1
    r2 += v2*vt*v2
  }
  if r0 == 0.0 {
    return 0.0
  } else {
    return r0 / math.Sqrt(r1) / math.Sqrt(r2)
  }
}

/* -------------------------------------------------------------------------- */

func similarity(config Config, filenameModel, filenameFasta, filenameOut string, negate bool) {
  classifier := ImportKmerLr(&config, filenameModel)

  // remove negative entries
  if negate {
    for i := 0; i < classifier.Theta.Dim(); i++ {
      if v := classifier.Theta.ValueAt(i); v > 0 {
        classifier.Theta.At(i).SetValue(0.0)
      } else {
        classifier.Theta.At(i).SetValue(-v)
      }
    }
  } else {
    for i := 0; i < classifier.Theta.Dim(); i++ {
      if v := classifier.Theta.ValueAt(i); v < 0 {
        classifier.Theta.At(i).SetValue(0.0)
      }
    }
  }
  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data, _ := compile_test_data(config, kmersCounter, classifier.Kmers, classifier.Features, filenameFasta)
  data     = classifier.Transform.Apply(config, data)

  // allocate result
  result := make([][]float64, len(data))
  for i := 0; i < len(data); i++ {
    result[i] = make([]float64, len(data))
    // create sparse vector index
    data[i].ValueAt(1)
  }
  config.Pool.RangeJob(0, len(data), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    for j := i; j < len(data); j++ {
      result[i][j] = compute_similarity(config, classifier.Theta, data[i], data[j])
      result[j][i] = result[i][j]
    }
    return nil
  })
  write_similarity_matrix(config, result, filenameOut)
}

/* -------------------------------------------------------------------------- */

func main_similarity(config Config, args []string) {
  log.SetFlags(0)

  options := getopt.New()

  optNegate := options.BoolLong("negate",  0 , "take negative coefficients to form the inner product space")
  optHelp   := options.BoolLong("help",   'h', "print help")

  options.SetParameters("<MODEL.json> [<INPUT.fasta> [OUTPUT.table]]")
  options.Parse(args)

  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if len(options.Args()) < 1 || len(options.Args()) > 3 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }

  filenameModel := options.Args()[0]
  filenameFasta := ""
  filenameOut   := ""
  if len(options.Args()) >= 2 {
    filenameFasta = options.Args()[1]
  }
  if len(options.Args()) == 3 {
    filenameOut   = options.Args()[2]
  }
  similarity(config, filenameModel, filenameFasta, filenameOut, *optNegate)
}
