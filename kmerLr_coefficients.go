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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func kmer_abundance(data []ConstVector, i, label int) float64 {
  k := 0
  n := 0
  for j := 0; j < len(data); j++ {
    if data[j].ValueAt(data[j].Dim()-1) == float64(label) {
      n += 1
      if data[j].ValueAt(i+1) > 0.0 {
        k += 1
      }
    }
  }
  return float64(k)/float64(n)
}

/* -------------------------------------------------------------------------- */

func coefficients_sort_map(kmers KmerClassList, coefficients map[KmerClassId]float64) []KmerClass {
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

func coefficients_related(kmer KmerClass, graph KmerGraph, coefficients map[KmerClassId]float64) []KmerClass {
  related := graph.RelatedKmers(kmer.Elements[0])
  return coefficients_sort_map(related, coefficients)
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

func coefficients_print(kmers KmerClassList, k int) string {
  k1, k2 := CoeffIndex(len(kmers)).Sub2Ind(k)
  if k1 == k2 {
    return fmt.Sprintf("%v", kmers[k1])
  } else {
    return fmt.Sprintf("%v & %v", kmers[k1], kmers[k2])
  }
}

func coefficients_format(kmers KmerClassList, n int) string {
  m := 0
  for k := 0; k < n; k++ {
    if r := len(coefficients_print(kmers, k)); r > m {
      m = r
    }
  }
  return fmt.Sprintf("%%6d %%14e %%%dv ", m)
}

/* -------------------------------------------------------------------------- */

func coefficients(config Config, filename, filename_fg, filename_bg string, related, rescale bool) {
  classifier   := ImportKmerLr(config, filename)
  coefficients := NewAbsFloatInt(classifier.Theta.Dim()-1)
  coeffmap     := make(map[KmerClassId]float64)
  kmers        := classifier.Kmers
  graph        := KmerGraph{}

  // copy config from classifier
  config.KmerEquivalence = classifier.KmerLrAlphabet.KmerEquivalence
  config.Binarize        = classifier.Binarize
  config.Cooccurrence    = classifier.Cooccurrence

  data := []ConstVector{}
  if filename_fg != "" && filename_bg != "" {
    kmersCounter, err := NewKmerCounter(config.M, config.N, config.Complement, config.Reverse, config.Revcomp, config.MaxAmbiguous, config.Alphabet); if err != nil {
      log.Fatal(err)
    }
    data, kmers = compile_training_data(config, kmersCounter, classifier.Kmers, filename_fg, filename_bg)
  }

  // insert coefficients into the map
  if rescale && len(classifier.Transform.Sigma) > 0 {
    for i, v := range classifier.Theta.GetValues()[1:] {
      coefficients.a[i] = v*classifier.Transform.Sigma[i+1]
      coefficients.b[i] = i
    }
  } else {
    for i, v := range classifier.Theta.GetValues()[1:] {
      coefficients.a[i] = v
      coefficients.b[i] = i
    }
  }
  coefficients.SortReverse()

  // construct graph of related k-mers if required
  if related {
    for i := 0; i < coefficients.Len(); i++ {
      v := coefficients.a[i]
      k := coefficients.b[i]
      if k < len(kmers) {
        coeffmap[kmers[k].KmerClassId] = v
      }
    }
    if rel, err := NewKmerEquivalenceRelation(classifier.M, classifier.N, classifier.Complement, classifier.Reverse, classifier.Revcomp, classifier.MaxAmbiguous, classifier.Alphabet); err != nil {
      log.Fatal(err)
    } else {
      graph = NewKmerGraph(kmers, rel)
    }
  }
  format := coefficients_format(kmers, coefficients.Len())

  for i := 0; i < coefficients.Len(); i++ {
    v := coefficients.a[i]
    k := coefficients.b[i]
    if v == 0.0 {
      break
    }
    if len(data) > 0 {
      fmt.Printf("%6.2f%% ", kmer_abundance(data, k, 1)*100.0)
      fmt.Printf("%6.2f%% ", kmer_abundance(data, k, 0)*100.0)
    }
    k1, k2 := CoeffIndex(len(kmers)).Sub2Ind(k)
    if k1 == k2 {
      fmt.Printf(format, i+1, v, kmers[k1])
    } else {
      fmt.Printf(format, i+1, v, fmt.Sprintf("%s & %s", kmers[k1], kmers[k2]))
    }
    if related && k < len(kmers) {
      coefficients_print_related(kmers[k], graph, coeffmap)
    }
    fmt.Println()
  }
}

/* -------------------------------------------------------------------------- */

func main_coefficients(config Config, args []string) {
  options := getopt.New()

  optRelated := options.BoolLong("related",   0 ,  "print related coefficients")
  optRescale := options.BoolLong("rescale",   0 ,  "rescale coefficients to untransform data")
  optHelp    := options.BoolLong("help",     'h',  "print help")

  options.SetParameters("<MODEL.json> [<FOREGROUND.fa> <BACKGROUND.fa>]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 1 && len(options.Args()) != 3 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename    := options.Args()[0]
  filename_fg := ""
  filename_bg := ""
  if len(options.Args()) == 3 {
    filename_fg = options.Args()[1]
    filename_bg = options.Args()[2]
  }

  coefficients(config, filename, filename_fg, filename_bg, *optRelated, *optRescale)
}
