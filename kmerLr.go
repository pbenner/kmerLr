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

import . "github.com/pbenner/ngstat/config"
import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type Config struct {
  SessionConfig
  Lambda         float64
  Epsilon        float64
  Seed           int64
  Binarize       bool
  Complement     bool
  Reverse        bool
  Revcomp        bool
  MaxAmbiguous []int
  Alphabet       ComplementableAlphabet
  SaveTrace      bool
  MaxEpochs      int
}


/* -------------------------------------------------------------------------- */

var Version   string
var BuildTime string
var GitHash   string

func printVersion(writer io.Writer) {
  fmt.Fprintf(writer, "ModHMM (https://github.com/pbenner/autodiff)\n")
  fmt.Fprintf(writer, " - Version   : %s\n", Version)
  fmt.Fprintf(writer, " - Build time: %s\n", BuildTime)
  fmt.Fprintf(writer, " - Git Hash  : %s\n", GitHash)
}

/* -------------------------------------------------------------------------- */

func main() {
  log.SetFlags(0)

  config  := Config{SessionConfig: DefaultSessionConfig()}
  options := getopt.New()

  optThreads := options.    IntLong("threads", 't',  1, "number of threads")
  optHelp    := options.   BoolLong("help",    'h',     "print help")
  optGenConf := options.   BoolLong("genconf",  0 ,     "print default config file")
  optVerbose := options.CounterLong("verbose", 'v',     "verbose level [-v or -vv]")
  optVersion := options.   BoolLong("version",  0 ,     "print ModHMM version")

  options.SetParameters("<COMMAND>\n\n" +
    " Commands:\n" +
    "     coverage                 [stage 1]   - compute single-feature coverages from bam files\n" +
    "     compute-counts           [stage 1.1] - compute coverage counts used for quantile normalization\n" +
    "     estimate-single-feature  [stage 1.2] - estimate mixture distribution for single-feature\n" +
    "                                            enrichment analysis\n" +
    "     eval-single-feature      [stage 2]   - evaluate single-feature models\n" +
    "     eval-multi-feature       [stage 3]   - evaluate  multi-feature models\n" +
    "     eval-posterior-marginals [stage 5]   - compute posterior marginals of hidden states\n\n" +
    "     segmentation             [stage 4]   - compute genome segmentation\n\n" +
    " ModHMM commands are structured in stages. Executing a command also executes all commands with\n" +
    " lower stage number.\n\n" +
    " Plotting commands:\n" +
    "     plot-single-feature                  - plot fitted single-feature mixture distribution used\n" +
    "                                            for enrichment analysis\n" +
    " Peak calling commands:\n" +
    "     call-single-feature-peaks            - call peaks of single-feature enrichment analysis\n" +
    "     call-multi-feature-peaks             - call peaks of multi-feature classifications\n" +
    "     call-posterior-marginal-peaks        - call peaks of HMM marginal posterior tracks\n")
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
  if *optGenConf  {
    config.Export(os.Stdout)
    os.Exit(0)
  }
  if *optThreads < 1 {
    log.Fatalf("invalid number of threads `%d'", *optThreads)
  }
  if options.Lookup('t').Seen() {
    config.Threads = *optThreads
  }
  // command arguments
  if len(options.Args()) == 0 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  command := options.Args()[0]

  switch command {
  case "learn":
    main_learn(config, options.Args())
  default:
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
}
