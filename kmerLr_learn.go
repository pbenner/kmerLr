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
import   "math"
import   "os"
import   "sort"
import   "strconv"
import   "strings"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func learn_parameters(config Config, classifier *KmerLrEnsemble, data_train, data_val, data_test KmerDataSet, icv int, basename_out string) ([]*KmerLrEnsemble, [][]float64, []float64, []float64) {
  estimator := NewKmerLrEnsembleEstimator(config, classifier, icv)

  classifiers, predictions, loss_train, loss_test := estimator.Estimate(config, data_train, data_val, data_test)

  filename_json  := basename_out
  filename_trace := basename_out
  filename_path  := basename_out

  if icv != -1 {
    filename_json  = fmt.Sprintf("%s_cv%d", filename_json , icv+1)
    filename_trace = fmt.Sprintf("%s_cv%d", filename_trace, icv+1)
    filename_path  = fmt.Sprintf("%s_cv%d", filename_path , icv+1)
  }
  // export trace
  if config.SaveTrace {
    SaveTrace(config, filename_trace+".trace", estimator.GetTrace())
  }
  // export path
  if config.SavePath {
    SaveKmerPath(config, filename_path+".path", estimator.GetPath())
  }
  if len(classifiers) == 1 {
    // export models
    SaveModel(config, filename_json+".json", classifiers[0])
  } else {
    for i, classifier := range classifiers {
      // export models
      SaveModel(config, fmt.Sprintf("%s_%d.json", filename_json, config.LambdaAuto[i]), classifier)
    }
  }
  return classifiers, predictions, loss_train, loss_test
}

func learn_cv(config Config, classifier *KmerLrEnsemble, data KmerDataSet, basename_out string) {
  learnAndTestClassifiers := func(i int, data_train, data_val, data_test KmerDataSet) ([][]float64, []float64, []float64) {
    _, predictions, loss_train, loss_test := learn_parameters(config, classifier, data_train, data_val, data_test, i, basename_out)
    return predictions, loss_train, loss_test
  }
  cvrs := crossvalidation(config, data, learnAndTestClassifiers)

  if len(cvrs) == 1 {
    SaveCrossvalidation    (config, fmt.Sprintf("%s.table"     , basename_out), cvrs[0])
    SaveCrossvalidationLoss(config, fmt.Sprintf("%s_loss.table", basename_out), cvrs[0])
  } else {
    for i, cvr := range cvrs {
      SaveCrossvalidation    (config, fmt.Sprintf("%s_%d.table"     , basename_out, config.LambdaAuto[i]), cvr)
      SaveCrossvalidationLoss(config, fmt.Sprintf("%s_%d_loss.table", basename_out, config.LambdaAuto[i]), cvr)
    }
  }
}

func learn(config Config, classifier *KmerLrEnsemble, filename_json, filename_fg, filename_bg, basename_out string) {
  if filename_json != "" {
    classifier = ImportKmerLrEnsemble(config, filename_json)
  }
  // do not use classifier.GetKmerCounter() since we do not want to fix the set of kmers!
  kmersCounter, err := NewKmerCounter(classifier.M, classifier.N, classifier.Complement, classifier.Reverse, classifier.Revcomp, classifier.MaxAmbiguous, classifier.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data := compile_training_data(config, kmersCounter, nil, nil, true, classifier.Binarize, filename_fg, filename_bg)
  kmersCounter = nil

  if len(data.Data) == 0 {
    log.Fatal("Error: no training data given")
  }
  // create index for sparse data
  for i, _ := range data.Data {
    data.Data[i].(SparseConstFloat64Vector).CreateIndex()
  }
  learn_cv(config, classifier, data, basename_out)
}

/* -------------------------------------------------------------------------- */

func main_learn(config Config, args []string) {
  options := getopt.New()

  // alphabet options
  optAlphabet        := options. StringLong("alphabet",           0 , "nucleotide", "nucleotide, gapped-nucleotide, or iupac-nucleotide")
  optBinarize        := options.   BoolLong("binarize",           0 ,               "binarize k-mer counts")
  optComplement      := options.   BoolLong("complement",         0 ,               "consider complement sequences")
  optCooccurrence    := options.   BoolLong("co-occurrence",      0 ,               "model k-mer co-occurrences")
  optMaxAmbiguous    := options. StringLong("max-ambiguous",      0 ,         "-1", "maxum number of ambiguous positions (either a scalar to set a global maximum or a comma separated list of length MAX-K-MER-LENGTH-MIN-K-MER-LENGTH+1)")
  optReverse         := options.   BoolLong("reverse",            0 ,               "consider reverse sequences")
  optRevcomp         := options.   BoolLong("revcomp",            0 ,               "consider reverse complement sequences")
  // other options
  optBalance         := options.   BoolLong("balance",            0 ,               "set class weights so that the data set is balanced")
  optLambda          := options. StringLong("lambda",             0 ,        "NaN", "set fixed regularization strength")
  optLambdaAuto      := options. StringLong("lambda-auto",        0 ,          "0", "comma separated list of integers specifying the number of features to select; for each value a separate classifier is estimated")
  optMaxFeatures     := options.    IntLong("max-features",       0 ,            0, "maximum number of features when a fixed lambda is set")
  optCopreselection  := options.    IntLong("co-preselection",    0 ,            0, "pre-select a subset of k-mers for co-occurrence modeling")
  optEnsembleSize    := options.    IntLong("ensemble-size",      0 ,            1, "estimate ensemble classifier")
  optEnsembleSummary := options. StringLong("ensemble-summary",   0 ,       "mean", "summary for classifier predictions [mean (default), product]")
  optMaxEpochs       := options.    IntLong("max-epochs",         0 ,            0, "maximum number of epochs")
  optMaxIterations   := options.    IntLong("max-iterations",     0 ,            0, "maximum number of iterations")
  optMaxSamples      := options.    IntLong("max-samples",        0 ,            0, "maximum number of samples")
  optEpsilon         := options. StringLong("epsilon",            0 ,       "0e-0", "optimization tolerance level for parameters")
  optEpsilonLambda   := options. StringLong("epsilon-lambda",     0 ,       "0e-0", "optimization tolerance level for lambda parameter")
  optEpsilonLoss     := options. StringLong("epsilon-loss",       0 ,       "1e-8", "optimization tolerance level for loss function")
  optSaveTrace       := options.   BoolLong("save-trace",         0 ,               "save trace to file")
  optSavePath        := options.   BoolLong("save-path",          0 ,               "save regularization path to file")
  optEvalLoss        := options.   BoolLong("eval-loss",          0 ,               "evaluate loss function after each epoch")
  optDataTransform   := options. StringLong("data-transform",     0 ,          "",  "transform data before training classifier [none (default), standardizer (preferred for dense data), variance-scaler (preferred for sparse data), max-abs-scaler, mean-scaler]")
  optKFoldCV         := options.    IntLong("k-fold-cv",          0 ,            1, "perform k-fold cross-validation")
  optValidationSize  := options. StringLong("validation-size",    0 ,        "0.0", "fraction of training data that should be used for validation [0.0 (default)]")
  optScaleStepSize   := options. StringLong("scale-step-size",    0 ,        "1.0", "scale standard step-size")
  optPenaltyFree     := options.   BoolLong("penalty-free",       0 ,               "re-estimate parameters without penalty after feature selection")
  optAdaptStepSize   := options.   BoolLong("adaptive-step-size", 0 ,               "adaptive step size during optimization")
  optThreadsCV       := options.    IntLong("threads-cv",         0 ,            1, "number of threads for cross-validation")
  optThreadsSaga     := options.    IntLong("threads-saga",       0 ,            1, "number of threads for SAGA algorithm")
  optThreadsLR       := options.    IntLong("threads-lr",         0 ,            1, "number of threads for evaluating the logistic loss and gradient")
  optHelp            := options.   BoolLong("help",              'h',               "print help")

  options.SetParameters("<<M> <N>|<MODEL.json>> <FOREGROUND.fa> <BACKGROUND.fa> <BASENAME_RESULT>")
  options.Parse(args)

  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 4 && len(options.Args()) != 5 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  classifier   := &KmerLrEnsemble{}
  filename_in  := ""
  filename_fg  := ""
  filename_bg  := ""
  basename_out := ""
  if len(options.Args()) == 5 {
    if m, err := strconv.ParseInt(options.Args()[0], 10, 64); err != nil {
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    } else {
      classifier.M = int(m)
    }
    if n, err := strconv.ParseInt(options.Args()[1], 10, 64); err != nil {
      options.PrintUsage(os.Stderr)
      os.Exit(1)
    } else {
      classifier.N = int(n)
    }
    if classifier.M < 1 || classifier.N < classifier.M {
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
  // parse classifier options
  //////////////////////////////////////////////////////////////////////////////
  classifier.Binarize     = *optBinarize
  classifier.Cooccurrence = *optCooccurrence
  classifier.Complement   = *optComplement
  classifier.Summary      = *optEnsembleSummary
  classifier.Reverse      = *optReverse
  classifier.Revcomp      = *optRevcomp
  if alphabet, err := alphabet_from_string(*optAlphabet); err != nil {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  } else {
    classifier.Alphabet = alphabet
  }
  if fields := strings.Split(*optMaxAmbiguous, ","); len(fields) == 1 || len(fields) == int(classifier.M-classifier.N+1) {
    classifier.MaxAmbiguous = make([]int, len(fields))
    for i := 0; i < len(fields); i++ {
      if t, err := strconv.ParseInt(fields[i], 10, 64); err != nil {
        options.PrintUsage(os.Stderr)
        os.Exit(1)
      } else {
        classifier.MaxAmbiguous[i] = int(t)
      }
    }
  } else {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  switch *optEnsembleSummary {
  case "mean":
  case "max":
  case "min":
  case "product":
  case "":
  default:
    options.PrintUsage(os.Stdout)
    os.Exit(1)
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
  if s, err := strconv.ParseFloat(*optLambda, 64); err != nil {
    log.Fatal(err)
  } else {
    config.Lambda = s
  }
  if fields := strings.Split(*optLambdaAuto, ","); len(fields) == 0 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  } else {
    for _, str := range fields {
      if n, err := strconv.ParseInt(str, 10, 64); err != nil {
        log.Fatal(err)
      } else {
        config.LambdaAuto = append(config.LambdaAuto, int(n))
      }
    }
    if len(config.LambdaAuto) == 0 {
      options.PrintUsage(os.Stdout)
      os.Exit(1)
    }
    sort.Ints(config.LambdaAuto)
  }
  if !math.IsNaN(config.Lambda) && (len(config.LambdaAuto) != 1 || config.LambdaAuto[0] != 0) {
    log.Fatal("options --lambda and --lambda-auto are incompatible")
  }
  if *optKFoldCV < 1 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  }
  if s, err := strconv.ParseFloat(*optValidationSize, 64); err != nil {
    log.Fatal(err)
  } else {
    if s < 0.0 || s > 1.0 {
      options.PrintUsage(os.Stdout)
      os.Exit(1)
    }
    config.ValidationSize = s
  }
  if *optEnsembleSize < 1 {
    options.PrintUsage(os.Stdout)
    os.Exit(1)
  }
  if *optThreadsCV > 1 {
    config.PoolCV = threadpool.New(*optThreadsCV, 100)
  }
  if *optThreadsSaga > 1 {
    config.PoolSaga = threadpool.New(*optThreadsSaga, 100)
  }
  if *optThreadsLR > 1 {
    config.PoolLR = threadpool.New(*optThreadsLR, 100)
  }
  config.AdaptStepSize   = *optAdaptStepSize
  config.Balance         = *optBalance
  config.Copreselection  = *optCopreselection
  config.EnsembleSize    = *optEnsembleSize
  config.EvalLoss        = *optEvalLoss
  config.KFoldCV         = *optKFoldCV
  config.MaxFeatures     = *optMaxFeatures
  config.MaxEpochs       = *optMaxEpochs
  config.MaxIterations   = *optMaxIterations
  config.MaxSamples      = *optMaxSamples
  config.PenaltyFree     = *optPenaltyFree
  config.SaveTrace       = *optSaveTrace
  config.SavePath        = *optSavePath
  config.DataTransform   = *optDataTransform
  switch strings.ToLower(config.DataTransform) {
  case "":
  case "none":
  case "standardizer":
  case "variance-scaler":
  case "max-abs-scaler":
  case "mean-scaler":
  default:
    log.Fatal("invalid data transform")
    panic("internal error")
  }
  if config.EpsilonLoss != 0.0 {
    config.EvalLoss = true
  }
  if config.AdaptStepSize {
    config.EvalLoss = true
  }
  learn(config, classifier, filename_in, filename_fg, filename_bg, basename_out)
}
