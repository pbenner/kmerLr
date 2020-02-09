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
import   "sort"
import   "strconv"
import   "strings"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func learn_scores_parameters(config Config, data_train, data_test []ConstVector, labels []bool, classifier *ScoresLr, features FeatureIndices, icv int, basename_out string) ([]*ScoresLr, [][]float64) {
  // hook and trace
  var trace *Trace
  if config.SaveTrace {
    trace = &Trace{}
  }

  estimator := NewScoresLrEstimator(config, trace, icv, features, labels)
  if classifier != nil {
    estimator.SetParameters(classifier.GetParameters().CloneVector())
  }
  classifiers, predictions := estimator.Estimate(config, data_train, data_test, labels)

  filename_trace := fmt.Sprintf("%s.trace", basename_out)
  // export trace
  if config.SaveTrace {
    SaveTrace(config, filename_trace, trace)
  }
  for i, classifier := range classifiers {
    filename_json := fmt.Sprintf("%s_%d.json" , basename_out, config.LambdaAuto[i])
    // export models
    SaveModel(config, filename_json, classifier)
  }
  return classifiers, predictions
}

func learn_scores_cv(config Config, data []ConstVector, labels []bool, classifier *ScoresLr, features FeatureIndices, basename_out string) {
  learnAndTestClassifiers := func(i int, data_train, data_test []ConstVector, labels []bool) [][]float64 {
    basename_out := fmt.Sprintf("%s_%d", basename_out, i+1)
    _, predictions := learn_scores_parameters(config, data_train, data_test, labels, classifier, features, i, basename_out)
    return predictions
  }
  cvrs := scoresCrossvalidation(config, data, labels, learnAndTestClassifiers)

  for i, cvr := range cvrs {
    SaveCrossvalidation(config, fmt.Sprintf("%s_%d.table", basename_out, i), cvr)
  }
}

func learn_scores(config Config, filename_json, filename_fg, filename_bg, basename_out string) {
  var classifier *ScoresLr
  var features    FeatureIndices
  if filename_json != "" {
    classifier = ImportScoresLr(&config, filename_json)
    features   = classifier.Features
  }
  data, labels := compile_training_data_scores(config, FeatureIndices{}, filename_fg, filename_bg)

  if len(data) == 0 {
    log.Fatal("Error: no training data given")
  }
  // create index for sparse data
  for i, _ := range data {
    data[i].(SparseConstRealVector).CreateIndex()
  }
  if config.KFoldCV <= 1 {
    learn_scores_parameters(config, data, nil, labels, classifier, features, -1, basename_out)
  } else {
    learn_scores_cv(config, data, labels, classifier, features, basename_out)
  }
}

/* -------------------------------------------------------------------------- */

func main_learn_scores(config Config, args []string) {
  options := getopt.New()

  optLambdaAuto      := options. StringLong("lambda-auto",      0 ,          "0", "select lambda automatically so that [value] coefficients are non-zero")
  optBalance         := options.   BoolLong("balance",          0 ,               "set class weights so that the data set is balanced")
  optCooccurrence    := options.   BoolLong("co-occurrence",    0 ,               "model co-occurrences")
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

  options.SetParameters("[MODEL.json] <FOREGROUND.table> <BACKGROUND.table> <BASENAME_RESULT>")
  options.Parse(args)

  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 3 && len(options.Args()) != 4 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_in  := ""
  filename_fg  := ""
  filename_bg  := ""
  basename_out := ""
  if len(options.Args()) == 3 {
    filename_fg  = options.Args()[0]
    filename_bg  = options.Args()[1]
    basename_out = options.Args()[2]
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
    sort.Ints(config.LambdaAuto)
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
  config.Cooccurrence    = *optCooccurrence
  config.KFoldCV         = *optKFoldCV
  config.EvalLoss        = *optEvalLoss
  config.MaxEpochs       = *optMaxEpochs
  config.MaxIterations   = *optMaxIterations
  config.SaveTrace       = *optSaveTrace
  config.NoNormalization = *optNoNormalization
  if config.EpsilonLoss != 0.0 {
    config.EvalLoss = true
  }
  learn_scores(config, filename_in, filename_fg, filename_bg, basename_out)
}
