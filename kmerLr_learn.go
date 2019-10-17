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
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func normalize_data(config Config, data []ConstVector) Transform {
  t := Transform{}
  t.TransformFit  (config, data)
  t.TransformApply(config, data)
  return t
}

/* -------------------------------------------------------------------------- */

func learn_parameters(config Config, data_train, data_test []ConstVector, labels []bool, classifier *KmerLr, kmers KmerClassList, features FeatureIndices, icv int, t Transform, basename_out string) VectorPdf {
  // hook and trace
  var trace *Trace
  if config.SaveTrace || config.EpsilonVar != 0.0 {
    trace = &Trace{}
  }
  if config.Omp != 0 {
    estimator := NewKmerLrOmpEstimator(config, kmers, trace, icv, data_train, labels, t)
    if classifier != nil {
      estimator.SetParameters(classifier.GetParameters().CloneVector())
    }
    classifier = estimator.Estimate(config, data_train, labels)
  } else
  if config.Rprop {
    estimator := NewKmerLrRpropEstimator(config, kmers, trace, icv, data_train, labels, t)
    if classifier != nil {
      estimator.SetParameters(classifier.GetParameters().CloneVector())
    }
    classifier = estimator.Estimate(config, data_train, labels)
  } else {
    estimator := NewKmerLrEstimator(config, kmers, trace, icv, data_train, features, labels, t)
    if classifier != nil {
      estimator.SetParameters(classifier.GetParameters().CloneVector())
    }
    classifier = estimator.Estimate(config, data_train, data_test, labels)
  }
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

func learn_cv(config Config, data []ConstVector, labels []bool, classifier *KmerLr, kmers KmerClassList, features FeatureIndices, t Transform, basename_out string) {
  learnClassifier := func(i int, data_train, data_test []ConstVector, labels []bool) VectorPdf {
    basename_out := fmt.Sprintf("%s_%d", basename_out, i+1)
    return learn_parameters(config, data_train, data_test, labels, classifier, kmers, features, i, t, basename_out)
  }
  testClassifier := func(i int, data []ConstVector, classifier VectorPdf) []float64 {
    return predict_data(config, data, classifier)
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

  // normalize data for faster convergence
  t := Transform{}
  if classifier == nil {
    if !config.NoNormalization {
      t = normalize_data(config, data)
    }
  } else {
    t = classifier.Transform
  }
  if config.KFoldCV <= 1 {
    learn_parameters(config, data, nil, labels, classifier, kmers, features, -1, t, basename_out)
  } else {
    learn_cv(config, data, labels, classifier, kmers, features, t, basename_out)
  }
}

/* -------------------------------------------------------------------------- */

func main_learn(config Config, args []string) {
  options := getopt.New()

  optAlphabet        := options. StringLong("alphabet",         0 , "nucleotide", "nucleotide, gapped-nucleotide, or iupac-nucleotide")
  optLambda          := options. StringLong("lambda",           0 ,        "0.0", "regularization strength (L1)")
  optLambdaAuto      := options.    IntLong("lambda-auto",      0 ,            0, "select lambda automatically so that [value] coefficients are non-zero")
  optLambdaEta       := options. StringLong("lambda-eta",       0 ,    "1.1:0.9", "auto lambda eta parameter [default: 1.1:0.9]")
  optLambdaMax       := options. StringLong("lambda-max",       0 ,        "0.0", "maximum lambda value")
  optBalance         := options.   BoolLong("balance",          0 ,               "set class weights so that the data set is balanced")
  optBinarize        := options.   BoolLong("binarize",         0 ,               "binarize k-mer counts")
  optCooccurrence    := options.    IntLong("co-occurrence",    0 ,           -1, "begin k-mer co-occurrences modeling when approximately [value] coefficients are non-zero")
  optComplement      := options.   BoolLong("complement",       0 ,               "consider complement sequences")
  optReverse         := options.   BoolLong("reverse",          0 ,               "consider reverse sequences")
  optRevcomp         := options.   BoolLong("revcomp",          0 ,               "consider reverse complement sequences")
  optMaxAmbiguous    := options. StringLong("max-ambiguous",    0 ,         "-1", "maxum number of ambiguous positions (either a scalar to set a global maximum or a comma separated list of length MAX-K-MER-LENGTH-MIN-K-MER-LENGTH+1)")
  optMaxEpochs       := options.    IntLong("max-epochs",       0 ,            0, "maximum number of epochs")
  optEpsilon         := options. StringLong("epsilon",          0 ,       "1e-4", "optimization tolerance level for parameters")
  optEpsilonLoss     := options. StringLong("epsilon-loss",     0 ,       "0e-0", "optimization tolerance level for loss function")
  optEpsilonVar      := options. StringLong("epsilon-var",      0 ,       "0e-0", "optimization tolerance level for variance of the number of components")
  optSaveTrace       := options.   BoolLong("save-trace",       0 ,               "save trace to file")
  optEvalLoss        := options.   BoolLong("eval-loss",        0 ,               "evaluate loss function after each epoch")
  optNoNormalization := options.   BoolLong("no-normalization", 0 ,               "do not normalize data")
  optKFoldCV         := options.    IntLong("k-fold-cv",        0 ,            1, "perform k-fold cross-validation")
  optScaleStepSize   := options. StringLong("scale-step-size",  0 ,        "1.0", "scale standard step-size")
  optRprop           := options.   BoolLong("rprop",            0 ,               "use rprop for optimization")
  optRpropStepSize   := options. StringLong("rprop-step-size",  0 ,        "0.0", "rprop initial step size [default: 0.0 (auto)]")
  optRpropEta        := options. StringLong("rprop-eta",        0 ,    "1.2:0.8", "rprop eta parameter [default: 1.2:0.8]")
  optOmp             := options.    IntLong("omp",              0 ,            0, "use OMP to select subset of features")
  optOmpIterations   := options.    IntLong("omp-iterations",   0 ,            1, "number of OMP iterations")
  optPrune           := options.    IntLong("prune",            0 ,            0, "prune parameter space if less than [value]% coefficients are non-zero")
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
  if *optLambdaAuto != 0 && *optRprop {
    log.Fatal("rprop does not support automatic regularization strength")
  }
  if s, err := strconv.ParseFloat(*optEpsilon, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Epsilon = s
  }
  if s, err := strconv.ParseFloat(*optEpsilonLoss, 64); err != nil {
    log.Fatal(err)
  } else {
    config.EpsilonLoss = s
  }
  if s, err := strconv.ParseFloat(*optEpsilonVar, 64); err != nil {
    log.Fatal(err)
  } else {
    config.EpsilonVar = s
  }
  if v, err := strconv.ParseFloat(*optLambda, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Lambda = v
  }
  if v, err := strconv.ParseFloat(*optLambdaMax, 64); err != nil {
    log.Fatal(err)
  } else {
    config.LambdaMax = v
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
  if *optOmp < 0 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  }
  if *optOmpIterations < 1 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  }
  if v, err := strconv.ParseFloat(*optRpropStepSize, 64); err != nil {
    log.Fatal(err)
  } else {
    config.RpropStepSize = v
  }
  if eta := strings.Split(*optRpropEta, ":"); len(eta) != 2 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  } else {
    v1, err := strconv.ParseFloat(eta[0], 64); if err != nil {
      log.Fatal(err)
    }
    v2, err := strconv.ParseFloat(eta[1], 64); if err != nil {
      log.Fatal(err)
    }
    config.RpropEta = []float64{v1, v2}
  }
  if eta := strings.Split(*optLambdaEta, ":"); len(eta) != 2 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  } else {
    v1, err := strconv.ParseFloat(eta[0], 64); if err != nil {
      log.Fatal(err)
    }
    v2, err := strconv.ParseFloat(eta[1], 64); if err != nil {
      log.Fatal(err)
    }
    config.LambdaEta = [2]float64{v1, v2}
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
  config.SaveTrace       = *optSaveTrace
  config.NoNormalization = *optNoNormalization
  config.Omp             = *optOmp
  config.OmpIterations   = *optOmpIterations
  config.Rprop           = *optRprop
  config.Prune           = *optPrune
  if config.EpsilonLoss != 0.0 {
    config.EvalLoss = true
  }
  if config.Prune != 0 && config.Lambda != 0.0 && config.LambdaAuto == 0.0 {
    log.Fatal("pruning requires automatic lambda mode")
  }
  if config.Rprop && config.Cooccurrence > 0 {
    log.Fatal("Rprop does not support delayed modeling of co-occurrences")
  }
  if config.Omp > 0 && config.Cooccurrence > 0 {
    log.Fatal("Omp does not support delayed modeling of co-occurrences")
  }

  learn(config, filename_in, filename_fg, filename_bg, basename_out)
}
