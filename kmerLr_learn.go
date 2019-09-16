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

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func normalize_data(config Config, data []ConstVector) Transform {
  PrintStderr(config, 1, "Normalizing data... ")
  t := Transform{}
  t.TransformFit  (data)
  t.TransformApply(data)
  PrintStderr(config, 1, "done\n")
  return t
}

/* -------------------------------------------------------------------------- */

func learn_parameters(config Config, data []ConstVector, classifier *KmerLr, kmers KmerClassList, icv int, t Transform, basename_out string) VectorPdf {
  // hook and trace
  var trace *Trace
  if config.SaveTrace || config.EpsilonVar != 0.0 {
    trace = &Trace{}
  }
  if config.Omp != 0 {
    estimator := NewKmerLrOmpEstimator(config, kmers)
    estimator.Hook = NewHook(config, trace, icv, data, &estimator.LogisticRegression)
    if classifier != nil {
      estimator.SetParameters(classifier.GetParameters())
    }
    classifier = estimator.Estimate(config, data)
  } else
  if config.Rprop {
    estimator := NewKmerLrRpropEstimator(config, kmers)
    estimator.Hook = NewRpropHook(config, trace, icv, data, estimator)
    if classifier != nil {
      estimator.SetParameters(classifier.GetParameters())
    }
    classifier = estimator.Estimate(config, data)
  } else
  if config.Hybrid != 0 {
    maxEpochs := config.MaxEpochs
    if config.MaxEpochs == 0 || config.MaxEpochs >= config.Hybrid {
      config.MaxEpochs = config.Hybrid
    }
    estimator1 := NewKmerLrRpropEstimator(config, kmers)
    estimator1.Hook = NewRpropHook(config, trace, icv, data, estimator1)
    if classifier != nil {
      estimator1.SetParameters(classifier.GetParameters())
    }
    classifier = estimator1.Estimate(config, data)
    if maxEpochs == 0 {
      config.MaxEpochs = 0
    } else {
      config.MaxEpochs = maxEpochs - config.Hybrid
      if config.MaxEpochs == 0 {
        config.MaxEpochs = -1
      }
    }
    if config.MaxEpochs >= 0 {
      estimator2 := NewKmerLrEstimator(config, kmers)
      estimator2.Hook = NewHook(config, trace, icv, data, &estimator2.LogisticRegression)
      estimator2.SetParameters(classifier.Theta)
      classifier = estimator2.Estimate(config, data)
    }
  } else {
    estimator := NewKmerLrEstimator(config, kmers)
    estimator.Hook = NewHook(config, trace, icv, data, &estimator.LogisticRegression)
    if classifier != nil {
      estimator.SetParameters(classifier.GetParameters())
    }
    classifier = estimator.Estimate(config, data)
  }
  filename_trace := fmt.Sprintf("%s.trace", basename_out)
  filename_json  := fmt.Sprintf("%s.json" , basename_out)
  // export trace
  if config.SaveTrace {
    SaveTrace(config, filename_trace, trace)
  }
  // set transform
  classifier.Transform = t
  // export model
  SaveModel(config, filename_json, classifier.Sparsify())

  return classifier
}

func learn_cv(config Config, data []ConstVector, classifier *KmerLr, kmers KmerClassList, kfold int, t Transform, basename_out string) {
  labels := getLabels(data)

  learnClassifier := func(i int, data []ConstVector) VectorPdf {
    basename_out := fmt.Sprintf("%s_%d", basename_out, i+1)
    return learn_parameters(config, data, classifier, kmers, i, t, basename_out)
  }
  testClassifier := func(i int, data []ConstVector, classifier VectorPdf) []float64 {
    return predict_labeled(config, data, classifier)
  }
  predictions, labels := crossvalidation(config, data, labels, kfold, learnClassifier, testClassifier)

  SaveCrossvalidation(config, fmt.Sprintf("%s.table", basename_out), predictions, labels)
}

func learn(config Config, kfold int, filename_json, filename_fg, filename_bg, basename_out string) {
  var classifier *KmerLr
  var kmers       KmerClassList
  if filename_json != "" {
    classifier = ImportKmerLr(config, filename_json)
    kmers      = classifier.Kmers
    // copy config from classifier
    config.KmerEquivalence = classifier.KmerLrAlphabet.KmerEquivalence
    config.Binarize        = classifier.Binarize
    config.Cooccurrence    = classifier.Cooccurrence
  }
  kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data, kmers := compile_training_data(config, kmersCounter, kmers, config.Binarize, filename_fg, filename_bg)
  kmersCounter = nil

  // normalize data for faster convergence
  t := Transform{}
  if !config.NoNormalization {
    t = normalize_data(config, data)
  }
  if kfold <= 1 {
    learn_parameters(config, data, classifier, kmers, -1, t, basename_out)
  } else {
    learn_cv(config, data, classifier, kmers, kfold, t, basename_out)
  }
}

/* -------------------------------------------------------------------------- */

func main_learn(config Config, args []string) {
  options := getopt.New()

  optAlphabet        := options. StringLong("alphabet",         0 , "nucleotide", "nucleotide, gapped-nucleotide, or iupac-nucleotide")
  optLambda          := options. StringLong("lambda",           0 ,        "0.0", "regularization strength (L1)")
  optBalance         := options.   BoolLong("balance",          0 ,               "set class weights so that the data set is balanced")
  optBinarize        := options.   BoolLong("binarize",         0 ,               "binarize k-mer counts")
  optCooccurrence    := options.   BoolLong("co-occurrence",    0 ,               "model k-mer co-occurrences")
  optComplement      := options.   BoolLong("complement",       0 ,               "consider complement sequences")
  optReverse         := options.   BoolLong("reverse",          0 ,               "consider reverse sequences")
  optRevcomp         := options.   BoolLong("revcomp",          0 ,               "consider reverse complement sequences")
  optMaxAmbiguous    := options. StringLong("max-ambiguous",    0 ,         "-1", "maxum number of ambiguous positions (either a scalar to set a global maximum or a comma separated list of length MAX-K-MER-LENGTH-MIN-K-MER-LENGTH+1)")
  optMaxEpochs       := options.    IntLong("max-epochs",       0 ,            0, "maximum number of epochs")
  optEpsilon         := options. StringLong("epsilon",          0 ,       "1e-4", "optimization tolerance level")
  optEpsilonVar      := options. StringLong("epsilon-var",      0 ,       "0e-0", "optimization tolerance level for the variance of the number of components")
  optSaveTrace       := options.   BoolLong("save-trace",       0 ,               "save trace to file")
  optEvalLoss        := options.   BoolLong("eval-loss",        0 ,               "evaluate loss function after each epoch")
  optNoNormalization := options.   BoolLong("no-normalization", 0 ,               "do not normalize data")
  optKFoldCV         := options.    IntLong("k-fold-cv",        0 ,            1, "perform k-fold cross-validation")
  optScaleStepSize   := options. StringLong("scale-step-size",  0 ,        "1.0", "scale standard step-size")
  optRprop           := options.   BoolLong("rprop",            0 ,               "use rprop for optimization")
  optRpropStepSize   := options. StringLong("rprop-step-size",  0 ,        "0.0", "rprop initial step size [default: 0.0 (auto)]")
  optRpropEta        := options. StringLong("rprop-eta",        0 ,    "1.2:0.8", "rprop eta parameter [default: 1.2:0.8]")
  optHybrid          := options.    IntLong("hybrid",           0 ,            0, "use rprop for n iterations and then switch to saga")
  optOmp             := options.    IntLong("omp",              0 ,            0, "use OMP to select subset of features")
  optOmpIterations   := options.    IntLong("omp-iterations",   0 ,            1, "number of OMP iterations")
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
  config.Balance         = *optBalance
  config.Binarize        = *optBinarize
  config.Complement      = *optComplement
  config.Cooccurrence    = *optCooccurrence
  config.Reverse         = *optReverse
  config.Revcomp         = *optRevcomp
  config.EvalLoss        = *optEvalLoss
  config.MaxEpochs       = *optMaxEpochs
  config.SaveTrace       = *optSaveTrace
  config.NoNormalization = *optNoNormalization
  config.Omp             = *optOmp
  config.OmpIterations   = *optOmpIterations
  config.Rprop           = *optRprop
  config.Hybrid          = *optHybrid

  learn(config, *optKFoldCV, filename_in, filename_fg, filename_bg, basename_out)
}
