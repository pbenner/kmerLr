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

import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func coefficients_sort(coefficients []float64) []int {
  r  := FloatInt{}
  r.a = make([]float64, len(coefficients))
  r.b = make([]int    , len(coefficients))
  for i, v := range coefficients {
    r.a[i] = math.Abs(v)
    r.b[i] = i
  }
  sort.Sort(sort.Reverse(r))
  return r.b
}

func coefficients_related(coefficients []float64, kmersCounter KmersCounter, k int) []int {
  re := kmersCounter.RelatedKmers(k)
  r  := FloatInt{}
  r.a = make([]float64, len(re))
  r.b = make([]int    , len(re))
  for i, kr := range re {
    r.a[i] = math.Abs(coefficients[kr])
    r.b[i] = kr
  }
  sort.Sort(sort.Reverse(r))
  return r.b
}

func coefficients_print(coefficients []float64, kmersCounter KmersCounter, indices []int) {
  first := true
  for _, i := range indices {
    if coefficients[i] != 0.0 {
      if !first {
        fmt.Printf(",")
      } else {
        first = false
      }
      fmt.Printf("%s:%e", kmersCounter.KmerName(i), coefficients[i])
    }
  }
}

func coefficients_print_related(coefficients []float64, kmersCounter KmersCounter, k int) {
  r := coefficients_related(coefficients, kmersCounter, k)
  coefficients_print(coefficients, kmersCounter, r)
}

/* -------------------------------------------------------------------------- */

func coefficients(config Config, filename string, related bool) {
  classifier   := ImportKmerLr(config, filename)
  coefficients := classifier.Theta.GetValues()

  kmersCounter, err := NewKmersCounter(classifier.M, classifier.N, classifier.Complement, classifier.Reverse, classifier.Revcomp, classifier.MaxAmbiguous, classifier.Alphabet); if err != nil {
    log.Fatal(err)
  }
  for i, k := range coefficients_sort(coefficients) {
    if coefficients[k] != 0.0 {
      fmt.Printf("%6d %14e %20s ", i+1, coefficients[k], kmersCounter.KmerName(k))
      if related {
        coefficients_print_related(coefficients, kmersCounter, k)
      }
      fmt.Println()
    }
  }
}

/* -------------------------------------------------------------------------- */

func main_coefficients(config Config, args []string) {
  options := getopt.New()

  optRelated := options.BoolLong("related",   0 ,  "print related coefficients")
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

  coefficients(config, filename, *optRelated)
}
