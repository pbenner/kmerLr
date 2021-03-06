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

func compute_similarity(config Config, theta []float64, x1, x2 ConstVector) float64 {
  r0 := 0.0
  r1 := 0.0
  r2 := 0.0
  for i := 0; i < len(theta); i++ {
    vt := theta[i]
    v1 := x1.Float64At(i)
    v2 := x2.Float64At(i)
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
  classifier := ImportKmerLrEnsemble(config, filenameModel).GetComponent(0)
  counter    := classifier.GetKmerCounter()

  // remove negative entries
  if negate {
    for i := 0; i < len(classifier.Theta); i++ {
      if v := classifier.Theta[i]; v > 0 {
        classifier.Theta[i] = 0.0
      } else {
        classifier.Theta[i] = -v
      }
    }
  } else {
    for i := 0; i < len(classifier.Theta); i++ {
      if v := classifier.Theta[i]; v < 0 {
        classifier.Theta[i] = 0.0
      }
    }
  }
  data := compile_test_data(config, counter, nil, nil, false, classifier.Binarize, filenameFasta)
  classifier.Transform.Apply(config, data.Data)

  // allocate result
  result := make([][]float64, len(data.Data))
  for i := 0; i < len(data.Data); i++ {
    result[i] = make([]float64, len(data.Data))
    // create sparse vector index
    data.Data[i].Float64At(1)
  }
  config.Pool.RangeJob(0, len(data.Data), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    config := config; config.Pool = pool

    for j := i; j < len(data.Data); j++ {
      result[i][j] = compute_similarity(config, classifier.Theta, data.Data[i], data.Data[j])
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
