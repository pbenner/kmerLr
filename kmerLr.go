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
import   "io"
import   "log"
import   "os"

import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type Config struct {
  Balance         bool
  Copreselection  int
  Lambda          float64
  LambdaAuto    []int
  MaxFeatures     int
  EnsembleSize    int
  EnsembleSummary string
  Epsilon         float64
  EpsilonLambda   float64
  EpsilonLoss     float64
  Header          bool
  KFoldCV         int
  AdaptStepSize   bool
  StepSizeFactor  float64
  Seed            int64
  SaveTrace       bool
  TraceFilename   string
  EvalLoss        bool
  MaxEpochs       int
  MaxIterations   int
  MaxSamples      int
  DataTransform   string
  Pool            threadpool.ThreadPool
  PoolCV          threadpool.ThreadPool
  PoolSaga        threadpool.ThreadPool
  PoolLR          threadpool.ThreadPool
  Verbose         int
}

/* -------------------------------------------------------------------------- */

var Version   string
var BuildTime string
var GitHash   string

func printVersion(writer io.Writer) {
  fmt.Fprintf(writer, "ModHMM (https://github.com/pbenner/kmerLr)\n")
  fmt.Fprintf(writer, " - Version   : %s\n", Version)
  fmt.Fprintf(writer, " - Build time: %s\n", BuildTime)
  fmt.Fprintf(writer, " - Git Hash  : %s\n", GitHash)
}

/* -------------------------------------------------------------------------- */

func main() {
  log.SetFlags(0)

  config  := Config{}
  options := getopt.New()

  optType    := options. StringLong("type",     0 ,  "kmerLr", "classifier type [kmerLr, scoresLr]")
  optThreads := options.    IntLong("threads",  0 ,         1, "number of threads")
  optSeed    := options.    IntLong("seed",     0 ,         1, "seed for the random number generateor")
  optHelp    := options.   BoolLong("help",    'h',            "print help")
  optVerbose := options.CounterLong("verbose", 'v',            "verbose level [-v or -vv]")
  optVersion := options.   BoolLong("version",  0 ,            "print version")

  options.SetParameters("<COMMAND>\n\n" +
    " Commands:\n" +
    "     learn          - estimate logistic regression parameters\n" +
    "     loss           - compute logistic loss\n" +
    "     predict        - use an estimated model to predict labels\n" +
    "     combine        - combine estimated models\n" +
    "     coefficients   - pretty-print coefficients\n")
  options.Parse(os.Args)

  // command options
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optVersion {
    printVersion(os.Stdout)
    os.Exit(0)
  }
  if *optVerbose != 0 {
    config.Verbose = *optVerbose
  }
  if *optThreads < 1 {
    log.Fatalf("invalid number of threads `%d'", *optThreads)
  }
  if *optThreads > 1 {
    config.Pool = threadpool.New(*optThreads, 100)
  }
  config.Seed = int64(*optSeed)
  // command arguments
  if len(options.Args()) == 0 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  command := options.Args()[0]

  switch *optType {
  case "kmerLr":
    switch command {
    case "learn":
      main_learn(config, options.Args())
    case "loss":
      main_loss(config, options.Args())
    case "predict":
      main_predict(config, options.Args())
    case "predict-genomic":
      main_predict_genomic(config, options.Args())
    case "combine":
      main_combine(config, options.Args())
    case "coefficients":
      main_coefficients(config, options.Args())
    case "count-features":
      main_count_features(config, options.Args())
    case "export":
      main_export(config, options.Args())
    case "similarity":
      main_similarity(config, options.Args())
    default:
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    }
  case "scoresLr":
    switch command {
    case "learn":
      main_learn_scores(config, options.Args())
    case "loss":
      main_loss_scores(config, options.Args())
    case "predict":
      main_predict_scores(config, options.Args())
    case "combine":
      main_combine_scores(config, options.Args())
    case "coefficients":
      main_coefficients_scores(config, options.Args())
    case "similarity":
      panic("not implemented")
    default:
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    }
  default:
    log.Fatal("invalid classifier type")
  }
}
