/* Copyright (C) 2020 Philipp Benner
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

//import   "fmt"
import   "log"
import   "os"
import   "strconv"
import   "strings"

import . "github.com/pbenner/gonetics"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func export(config Config, classifier *KmerLrEnsemble, filename_json string, filename_seq, filename_out []string) {
  if filename_json != "" {
    classifier = ImportKmerLrEnsemble(config, filename_json)
  }
  // do not use classifier.GetKmerCounter() since we do not want to fix the set of kmers!
  kmersCounter, err := NewKmerCounter(classifier.M, classifier.N, classifier.Complement, classifier.Reverse, classifier.Revcomp, classifier.MaxAmbiguous, classifier.Alphabet); if err != nil {
    log.Fatal(err)
  }
  data := compile_data(config, kmersCounter, nil, nil, true, classifier.Binarize, filename_seq)
  kmersCounter = nil

  for i, _ := range data {
    export_kmers(config, filename_out[i], data[i])
  }
}

/* -------------------------------------------------------------------------- */

func main_export(config Config, args []string) {
  options := getopt.New()

  // alphabet options
  optAlphabet        := options. StringLong("alphabet",         0 , "nucleotide", "nucleotide, gapped-nucleotide, or iupac-nucleotide")
  optBinarize        := options.   BoolLong("binarize",         0 ,               "binarize k-mer counts")
  optComplement      := options.   BoolLong("complement",       0 ,               "consider complement sequences")
  optMaxAmbiguous    := options. StringLong("max-ambiguous",    0 ,         "-1", "maxum number of ambiguous positions (either a scalar to set a global maximum or a comma separated list of length MAX-K-MER-LENGTH-MIN-K-MER-LENGTH+1)")
  optReverse         := options.   BoolLong("reverse",          0 ,               "consider reverse sequences")
  optRevcomp         := options.   BoolLong("revcomp",          0 ,               "consider reverse complement sequences")
  // other options
  optHelp            := options.   BoolLong("help",            'h',               "print help")

  options.SetParameters("<<M> <N>|<MODEL.json>> <SEQUENCES_1.fa,SEQUENCES_2.fa,...> <OUPTUT_1.table,OUPTUT_2.table,...>")
  options.Parse(args)

  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 3 && len(options.Args()) != 4 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  classifier   := &KmerLrEnsemble{}
  filename_in  := ""
  filename_seq := ""
  filename_out := ""
  if len(options.Args()) == 4 {
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
    filename_seq = options.Args()[2]
    filename_out = options.Args()[3]
  } else {
    filename_in  = options.Args()[0]
    filename_seq = options.Args()[1]
    filename_out = options.Args()[2]
  }
  // parse classifier options
  //////////////////////////////////////////////////////////////////////////////
  classifier.Binarize     = *optBinarize
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
  filenames_1 := strings.Split(filename_seq, ",")
  filenames_2 := strings.Split(filename_out, ",")
  if len(filenames_1) != len(filenames_2) {
    options.PrintUsage(os.Stdout)
    os.Exit(0)    
  }
  export(config, classifier, filename_in, filenames_1, filenames_2)
}
