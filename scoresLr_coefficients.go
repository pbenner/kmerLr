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
import   "os"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func coefficients_print_scores(index []int, names []string, features FeatureIndices, k int) string {
  k1 := index[features[k][0]]
  k2 := index[features[k][1]]
  if len(names) == 0 {
    if k1 == k2 {
      return fmt.Sprintf("%d", k1+1)
    } else {
      return fmt.Sprintf("%d & %d", k1+1, k2+1)
    }
  } else {
    n1 := names[features[k][0]]
    n2 := names[features[k][1]]
    if k1 == k2 {
      return fmt.Sprintf("%s", n1)
    } else {
      return fmt.Sprintf("%s & %s", n1, n2)
    }
  }
}

func coefficients_format_scores(index []int, names []string, features FeatureIndices, coefficients AbsFloatInt) string {
  m := 0
  for k := 0; k < coefficients.Len(); k++ {
    if coefficients.a[k] == 0.0 {
      break
    }
    if r := len(coefficients_print_scores(index, names, features, coefficients.b[k])); r > m {
      m = r
    }
  }
  return fmt.Sprintf("%%6d %%14e %%%dv ", m)
}

/* -------------------------------------------------------------------------- */

func coefficients_scores_(config Config, classifier *ScoresLr, i_ int, rescale bool) {
  coefficients := NewAbsFloatInt(len(classifier.Theta)-1)
  features     := classifier.Features
  index        := classifier.Index
  names        := classifier.Names

  // insert coefficients into the map
  if rescale && len(classifier.Transform.Sigma) > 0 {
    for i, v := range classifier.Theta[1:] {
      coefficients.a[i] = v*classifier.Transform.Sigma[i+1]
      coefficients.b[i] = i
    }
  } else {
    for i, v := range classifier.Theta[1:] {
      coefficients.a[i] = v
      coefficients.b[i] = i
    }
  }
  coefficients.SortReverse()

  format := coefficients_format_scores(index, names, features, coefficients)

  fmt.Printf("Classifier %d:\n", i_)
  for i := 0; i < coefficients.Len(); i++ {
    v := coefficients.a[i]
    k := coefficients.b[i]
    if v == 0.0 {
      break
    }
    k1 := index[features[k][0]]
    k2 := index[features[k][1]]
    if len(names) == 0 {
      if k1 == k2 {
        fmt.Printf(format, i+1, v, k1+1)
      } else {
        fmt.Printf(format, i+1, v, fmt.Sprintf("%d & %d", k1+1, k2+1))
      }
    } else {
      n1 := names[features[k][0]]
      n2 := names[features[k][1]]
      if k1 == k2 {
        fmt.Printf(format, i+1, v, n1)
      } else {
        fmt.Printf(format, i+1, v, fmt.Sprintf("%s & %s", n1, n2))
      }
    }
    fmt.Println()
  }
}

func coefficients_scores(config Config, filename string, rescale bool) {
  classifier := ImportScoresLrEnsemble(config, filename)

  for i := 0; i < classifier.EnsembleSize(); i++ {
    coefficients_scores_(config, classifier.GetComponent(i), i, rescale)
  }
}

/* -------------------------------------------------------------------------- */

func main_coefficients_scores(config Config, args []string) {
  options := getopt.New()

  optRescale := options.BoolLong("rescale",   0 ,  "rescale coefficients to untransformed data")
  optHelp    := options.BoolLong("help",     'h',  "print help")

  options.SetParameters("<MODEL.json>")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 1 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename := options.Args()[0]

  coefficients_scores(config, filename, *optRescale)
}
