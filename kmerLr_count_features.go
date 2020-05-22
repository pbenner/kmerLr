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

import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func count_features(config Config, classifier *KmerLrEnsemble, filename_fg, filename_bg string) {
  // do not use classifier.GetKmerCounter() since we do not want to fix the set of kmers!
  kmersCounter, err := NewKmerCounter(classifier.M, classifier.N, classifier.Complement, classifier.Reverse, classifier.Revcomp, classifier.MaxAmbiguous, classifier.Alphabet); if err != nil {
    log.Fatal(err)
  }
  var data KmerDataSet
  if filename_bg == "" {
    data = compile_test_data(config, kmersCounter, nil, nil, classifier.Binarize, filename_fg)
  } else {
    data = compile_training_data(config, kmersCounter, nil, nil, classifier.Binarize, filename_fg, filename_bg)
  }
  fmt.Println(len(data.Kmers))
}

/* -------------------------------------------------------------------------- */

func main_count_features(config Config, args []string) {
  options := getopt.New()

  optAlphabet        := options. StringLong("alphabet",         0 , "nucleotide", "nucleotide, gapped-nucleotide, or iupac-nucleotide")
  optComplement      := options.   BoolLong("complement",       0 ,               "consider complement sequences")
  optCooccurrence    := options.   BoolLong("co-occurrence",    0 ,               "model k-mer co-occurrences")
  optMaxAmbiguous    := options. StringLong("max-ambiguous",    0 ,         "-1", "maxum number of ambiguous positions (either a scalar to set a global maximum or a comma separated list of length MAX-K-MER-LENGTH-MIN-K-MER-LENGTH+1)")
  optReverse         := options.   BoolLong("reverse",          0 ,               "consider reverse sequences")
  optRevcomp         := options.   BoolLong("revcomp",          0 ,               "consider reverse complement sequences")
  optHelp            := options.   BoolLong("help",            'h',               "print help")

  options.SetParameters("<M> <N> <FOREGROUND.fa> [BACKGROUND.fa]")
  options.Parse(args)

  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 3 && len(options.Args()) != 4 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  classifier   := &KmerLrEnsemble{}
  filename_fg  := ""
  filename_bg  := ""
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
  if len(options.Args()) == 3 {
    filename_fg  = options.Args()[2]
  } else {
    filename_fg  = options.Args()[2]
    filename_bg  = options.Args()[3]
  }
  // parse classifier options
  //////////////////////////////////////////////////////////////////////////////
  classifier.Cooccurrence = *optCooccurrence
  classifier.Complement   = *optComplement
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
  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  count_features(config, classifier, filename_fg, filename_bg)
}
