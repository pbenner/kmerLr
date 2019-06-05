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
import   "log"
import   "os"
import   "strconv"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func learn_parameters(config Config, data []ConstVector, n int, basename_out string) VectorPdf {
  // hook and trace
  var trace *Trace
  if config.SaveTrace {
    trace = &Trace{}
  }
  estimator  := NewKmerLrEstimator(config, n, NewHook(config, trace))
  classifier := estimator.Estimate(config, data)

  filename_trace := fmt.Sprintf("%s.trace.table", basename_out)
  filename_json  := fmt.Sprintf("%s.json"       , basename_out)
  // export trace
  if config.SaveTrace {
    PrintStderr(config, 1, "Exporting trace to `%s'... ", filename_trace)
    if err := trace.Export(filename_trace); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    }
    PrintStderr(config, 1, "done\n")
  }
  // export model
  PrintStderr(config, 1, "Exporting distribution to `%s'... ", filename_json)
  if err := ExportDistribution(filename_json, classifier); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")

  return classifier
}

func learn_cv(config Config, data []ConstVector, n, kfold int, basename_out string) {
  labels := getLabels(data)

  learnClassifier := func(i int, data []ConstVector) VectorPdf {
    basename_out := fmt.Sprintf("%s_%d", basename_out, i+1)
    return learn_parameters(config, data, n, basename_out)
  }
  testClassifier := func(i int, data []ConstVector, classifier VectorPdf) []float64 {
    return predict_labeled(config, classifier, data)
  }
  predictions, labels := crossvalidation(config, data, labels, kfold, learnClassifier, testClassifier)

  saveCrossvalidation(fmt.Sprintf("%s.table", basename_out), predictions, labels)
}

func learn(config Config, m, n, kfold int, filename_fg, filename_bg, basename_out string) {
  kmersCounter, err := NewKmersCounter(m, n, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data := compile_training_data(config, kmersCounter, config.Binarize, filename_fg, filename_bg)

  if kfold <= 1 {
    learn_parameters(config, data, kmersCounter.Length(), basename_out)
  } else {
    learn_cv(config, data, kmersCounter.Length(), kfold, basename_out)
  }
}

/* -------------------------------------------------------------------------- */

func main_learn(config Config, args []string) {
  log.SetFlags(0)

  options := getopt.New()

  optAlphabet   := options. StringLong("alphabet",   0 , "nucleotide", "nucleotide, gappend-nucleotide, or iupac-nucleotide")
  optLambda     := options. StringLong("lambda",     0 ,        "0.0", "regularization strength (L1)")
  optBinarize   := options.   BoolLong("binarize",   0 ,               "binarize k-mer counts")
  optComplement := options.   BoolLong("complement", 0 ,               "consider complement sequences")
  optReverse    := options.   BoolLong("reverse",    0 ,               "consider reverse sequences")
  optRevcomp    := options.   BoolLong("revcomp",    0 ,               "consider reverse complement sequences")
  optMaxEpochs  := options.    IntLong("max-epochs", 0 ,            1, "maximum number of epochs")
  optEpsilon    := options. StringLong("epsilon",    0 ,       "1e-4", "optimization tolerance level")
  optSaveTrace  := options.   BoolLong("save-trace", 0 ,               "save trace to file")
  optKFoldCV    := options.    IntLong("k-fold-cv",  0 ,            1, "perform k-fold cross-validation")
  optVerbose    := options.CounterLong("verbose",   'v',               "verbose level [-v or -vv]")
  optHelp       := options.   BoolLong("help",      'h',               "print help")

  options.SetParameters("<N> <M> <FOREGROUND.fa> <BACKGROUND.fa> <BASENAME_RESULT>")
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
  if s, err := strconv.ParseFloat(*optEpsilon, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Epsilon = s
  }
  if v, err := strconv.ParseFloat(*optLambda, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Lambda = v
  }
  if alphabet, err := alphabet_from_string(*optAlphabet); err != nil {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  } else {
    config.Alphabet = alphabet
  }
  if *optKFoldCV < 1 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  }
  config.Binarize      = *optBinarize
  config.Complement    = *optComplement
  config.Reverse       = *optReverse
  config.Revcomp       = *optRevcomp
  config.MaxEpochs     = *optMaxEpochs
  config.SaveTrace     = *optSaveTrace
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 5 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  n, err := strconv.ParseInt(options.Args()[0], 10, 64); if err != nil {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  m, err := strconv.ParseInt(options.Args()[1], 10, 64); if err != nil {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  if n < 1 || m < n {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  filename_fg  := options.Args()[2]
  filename_bg  := options.Args()[3]
  basename_out := options.Args()[4]

  learn(config, int(n), int(m), *optKFoldCV, filename_fg, filename_bg, basename_out)
}
