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

//import   "fmt"
import   "log"
import   "os"
import   "strings"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func combine(config Config, summary, filename_out string, filename_ins ...string) {
  classifiers := make([]*KmerLr, len(filename_ins))
  for i, filename_in := range filename_ins {
    classifiers[i] = ImportKmerLr(config, filename_in)
    if !classifiers[i].KmerLrAlphabet.KmerEquivalence.Equals(classifiers[0].KmerLrAlphabet.KmerEquivalence) {
      log.Fatalf("alphabet not consistent across classifiers")
    }
    if  classifiers[i].Binarize != classifiers[0].Binarize {
      log.Fatalf("data binarization is not consistent across classifiers")
    }
    if  classifiers[i].Cooccurrence != classifiers[0].Cooccurrence {
      log.Fatalf("data binarization is not consistent across classifiers")
    }
    if !classifiers[i].Transform.TransformEquals(classifiers[0].Transform) {
      log.Fatalf("data transform is not consistent across classifiers")
    }
  }
  r := classifiers[0].Clone()
  switch strings.ToLower(summary) {
  case "mean":
    r.Mean(classifiers)
  case "max":
    r.Max(classifiers)
  case "min":
    r.Min(classifiers)
  default:
    log.Fatalf("invalid summary statistic: %s", summary)
  }
  // export model
  SaveModel(config, filename_out, r)
}

/* -------------------------------------------------------------------------- */

func main_combine(config Config, args []string) {
  options := getopt.New()

  optSummary := options. StringLong("summary",    0 , "mean", "summary [mean (default), max, min]")
  optHelp    := options.   BoolLong("help",      'h',         "print help")

  options.SetParameters("<RESULT.json> <MODEL1.json> [<MODEL2.json>...]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) < 2 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_out := options.Args()[0]
  filename_ins := options.Args()[1:len(options.Args())]

  combine(config, *optSummary, filename_out, filename_ins...)
}
