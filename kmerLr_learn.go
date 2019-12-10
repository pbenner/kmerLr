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
import   "strings"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func learn_parameters(config Config, data_train, data_test []ConstVector, labels []bool, classifier *KmerLr, kmers KmerClassList, features FeatureIndices, t TransformFull, icv int, basename_out string) *KmerLr {
  // hook and trace
  var trace *Trace
  if config.SaveTrace {
    trace = &Trace{}
  }

  estimator := NewKmerLrEstimator(config, kmers, trace, icv, data_train, features, labels)
  if classifier != nil {
    estimator.SetParameters(classifier.GetParameters().CloneVector())
  }
  classifier = estimator.Estimate(config, data_train, data_test, labels, t)

  filename_trace := fmt.Sprintf("%s.trace", basename_out)
  filename_json  := fmt.Sprintf("%s.json" , basename_out)
  // export trace
  if config.SaveTrace {
    SaveTrace(config, filename_trace, trace)
  }
  // export model
  SaveModel(config, filename_json, classifier)

  return classifier
}

func learn_cv(config Config, data []ConstVector, labels []bool, classifier *KmerLr, kmers KmerClassList, features FeatureIndices, t TransformFull, basename_out string) {
  learnClassifier := func(i int, data_train, data_test []ConstVector, labels []bool) *KmerLr {
    basename_out := fmt.Sprintf("%s_%d", basename_out, i+1)
    return learn_parameters(config, data_train, data_test, labels, classifier, kmers, features, t, i, basename_out)
  }
  testClassifier := func(i int, data []ConstVector, classifier *KmerLr) []float64 {
    return classifier.Predict(config, data)
  }
  predictions, labels := crossvalidation(config, data, labels, learnClassifier, testClassifier)

  SaveCrossvalidation(config, fmt.Sprintf("%s.table", basename_out), predictions, labels)
}

func learn(config Config, filename_json, filename_fg, filename_bg, basename_out string) {
  var classifier *KmerLr
  var kmers       KmerClassList
  var features    FeatureIndices
  if filename_json != "" {
    classifier = ImportKmerLr(&config, filename_json)
    kmers      = classifier.Kmers
    features   = classifier.Features
  }
  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data, labels, kmers := compile_training_data(config, kmersCounter, kmers, features, filename_fg, filename_bg)
  kmersCounter = nil

  if len(data) == 0 {
    log.Fatal("Error: no training data given")
  }
  t := TransformFull{}
  // estimate transform on full data set so that all estimated
  // classifiers share the same transform
  if !config.NoNormalization {
    t.Fit(config, data)
  }
  // create index for sparse data
  for i, _ := range data {
    data[i].(SparseConstRealVector).CreateIndex()
  }
  if config.KFoldCV <= 1 {
    learn_parameters(config, data, nil, labels, classifier, kmers, features, t, -1, basename_out)
  } else {
    learn_cv(config, data, labels, classifier, kmers, features, t, basename_out)
  }
}

/* -------------------------------------------------------------------------- */

func main_learn(config Config, args []string) {
  options := getopt.New()

  optAlphabet        := options. StringLong("alphabet",         0 , "nucleotide", "nucleotide, gapped-nucleotide, or iupac-nucleotide")
  optLambdaAuto      := options.    IntLong("lambda-auto",      0 ,            0, "select lambda automatically so that [value] coefficients are non-zero")
  optBalance         := options.   BoolLong("balance",          0 ,               "set class weights so that the data set is balanced")
  optBinarize        := options.   BoolLong("binarize",         0 ,               "binarize k-mer counts")
  optCooccurrence    := options.   BoolLong("co-occurrence",    0 ,               "model k-mer co-occurrences")
  optComplement      := options.   BoolLong("complement",       0 ,               "consider complement sequences")
  optReverse         := options.   BoolLong("reverse",          0 ,               "consider reverse sequences")
  optRevcomp         := options.   BoolLong("revcomp",          0 ,               "consider reverse complement sequences")
  optMaxAmbiguous    := options. StringLong("max-ambiguous",    0 ,         "-1", "maxum number of ambiguous positions (either a scalar to set a global maximum or a comma separated list of length MAX-K-MER-LENGTH-MIN-K-MER-LENGTH+1)")
  optMaxEpochs       := options.    IntLong("max-epochs",       0 ,            0, "maximum number of epochs")
  optMaxIterations   := options.    IntLong("max-iterations",   0 ,            0, "maximum number of iterations")
  optEpsilon         := options. StringLong("epsilon",          0 ,       "0e-0", "optimization tolerance level for parameters")
  optEpsilonLambda   := options. StringLong("epsilon-lambda",   0 ,       "0e-0", "optimization tolerance level for lambda parameter")
  optEpsilonLoss     := options. StringLong("epsilon-loss",     0 ,       "1e-5", "optimization tolerance level for loss function")
  optSaveTrace       := options.   BoolLong("save-trace",       0 ,               "save trace to file")
  optEvalLoss        := options.   BoolLong("eval-loss",        0 ,               "evaluate loss function after each epoch")
  optNoNormalization := options.   BoolLong("no-normalization", 0 ,               "do not normalize data")
  optKFoldCV         := options.    IntLong("k-fold-cv",        0 ,            1, "perform k-fold cross-validation")
  optScaleStepSize   := options. StringLong("scale-step-size",  0 ,        "1.0", "scale standard step-size")
  optThreadsCV       := options.    IntLong("threads-cv",       0 ,            1, "number of threads for cross-validation")
  optThreadsSaga     := options.    IntLong("threads-saga",     0 ,            1, "number of threads for SAGA algorithm")
  optHelp            := options.   BoolLong("help",            'h',               "print help")

  options.SetParameters("<<M> <N>|<MODEL.json>> <FOREGROUND.fa> <BACKGROUND.fa> <BASENAME_RESULT>")
  options.Parse(args)

  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 4 && len(options.Args()) != 5 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_in  := ""
  filename_fg  := ""
  filename_bg  := ""
  basename_out := ""
  if len(options.Args()) == 5 {
    if m, err := strconv.ParseInt(options.Args()[0], 10, 64); err != nil {
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    } else {
      config.M = int(m)
    }
    if n, err := strconv.ParseInt(options.Args()[1], 10, 64); err != nil {
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    } else {
      config.N = int(n)
    }
    if config.M < 1 || config.N < config.M {
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    }
    filename_fg  = options.Args()[2]
    filename_bg  = options.Args()[3]
    basename_out = options.Args()[4]
  } else {
    filename_in  = options.Args()[0]
    filename_fg  = options.Args()[1]
    filename_bg  = options.Args()[2]
    basename_out = options.Args()[3]
  }
  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if s, err := strconv.ParseFloat(*optEpsilon, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Epsilon = s
  }
  if s, err := strconv.ParseFloat(*optEpsilonLambda, 64); err != nil {
    log.Fatal(err)
  } else {
    config.EpsilonLambda = s
  }
  if s, err := strconv.ParseFloat(*optEpsilonLoss, 64); err != nil {
    log.Fatal(err)
  } else {
    config.EpsilonLoss = s
  }
  if v, err := strconv.ParseFloat(*optScaleStepSize, 64); err != nil {
    log.Fatal(err)
  } else {
    config.StepSizeFactor = v
  }
  if fields := strings.Split(*optMaxAmbiguous, ","); len(fields) == 1 || len(fields) == int(config.M-config.N+1) {
    config.MaxAmbiguous = make([]int, len(fields))
    for i := 0; i < len(fields); i++ {
      if t, err := strconv.ParseInt(fields[i], 10, 64); err != nil {
        options.PrintUsage(os.Stderr)
        os.Exit(1)
      } else {
        config.MaxAmbiguous[i] = int(t)
      }
    }
  } else {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
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
  if *optThreadsCV > 1 {
    config.PoolCV = threadpool.New(*optThreadsCV, 100)
  }
  if *optThreadsSaga > 1 {
    config.PoolSaga = threadpool.New(*optThreadsSaga, 100)
  }
  config.Balance         = *optBalance
  config.Binarize        = *optBinarize
  config.Complement      = *optComplement
  config.Cooccurrence    = *optCooccurrence
  config.LambdaAuto      = *optLambdaAuto
  config.KFoldCV         = *optKFoldCV
  config.Reverse         = *optReverse
  config.Revcomp         = *optRevcomp
  config.EvalLoss        = *optEvalLoss
  config.MaxEpochs       = *optMaxEpochs
  config.MaxIterations   = *optMaxIterations
  config.SaveTrace       = *optSaveTrace
  config.NoNormalization = *optNoNormalization
  if config.EpsilonLoss != 0.0 {
    config.EvalLoss = true
  }
  learn(config, filename_in, filename_fg, filename_bg, basename_out)
}
