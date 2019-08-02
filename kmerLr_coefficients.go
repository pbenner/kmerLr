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

import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func coefficients_sort(kmers KmerClassList, coefficients map[KmerClassId]float64) []KmerClass {
  r  := FloatKmer{}
  r.a = []float64  (nil)
  r.b = []KmerClass(nil)
  for _, kmer := range kmers {
    r.a = append(r.a, math.Abs(coefficients[kmer.KmerClassId]))
    r.b = append(r.b, kmer)
  }
  r.SortReverse()
  return r.b
}

/* -------------------------------------------------------------------------- */

func coefficients_related(kmer KmerClass, graph KmerGraph, coefficients map[KmerClassId]float64) []KmerClass {
  related := graph.RelatedKmers(kmer.Elements[0])
  return coefficients_sort(related, coefficients)
}

func coefficients_print_related(kmer KmerClass, graph KmerGraph, coefficients map[KmerClassId]float64) {
  related := coefficients_related(kmer, graph, coefficients)
  first   := true
  for _, r := range related {
    if !first {
      fmt.Printf(",")
    } else {
      first = false
    }
    fmt.Printf("%s:%e", r, coefficients[r.KmerClassId])
  }
}

/* -------------------------------------------------------------------------- */

func coefficients_format(kmers KmerClassList) string {
  n := 0
  for _, kmer := range kmers {
    if r := len(kmer.String()); r > n {
      n = r
    }
  }
  return fmt.Sprintf("%%6d %%14e %%%ds ", n)
}

/* -------------------------------------------------------------------------- */

func coefficients(config Config, filename string, related bool) {
  classifier   := ImportKmerLr(config, filename)
  coefficients := make(map[KmerClassId]float64)
  kmers        := classifier.Kmers
  graph        := KmerGraph{}

  // insert coefficients into the map
  for i, v := range classifier.Theta.GetValues()[1:] {
    coefficients[kmers[i].KmerClassId] = v
  }
  // construct graph of related k-mers if required
  if related {
    if rel, err := NewKmerEquivalenceRelation(classifier.M, classifier.N, classifier.Complement, classifier.Reverse, classifier.Revcomp, classifier.MaxAmbiguous, classifier.Alphabet); err != nil {
      log.Fatal(err)
    } else {
      graph = NewKmerGraph(kmers, rel)
    }
  }
  format := coefficients_format(kmers)

  for i, kmer := range coefficients_sort(kmers, coefficients) {
    fmt.Printf(format, i+1, coefficients[kmer.KmerClassId], kmer.String())
    if related {
      coefficients_print_related(kmer, graph, coefficients)
    }
    fmt.Println()
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
