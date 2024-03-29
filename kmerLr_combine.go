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
import   "bufio"
import   "log"
import   "os"
import   "strings"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func stability_analysis(config Config, kmerLr *KmerLrEnsemble, filename string) {
  x, y := kmerLr.Stability()

  f, err := os.Create(filename)
  if err != nil {
    panic(err)
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  fmt.Fprintf(w, "stability mean\n")
  for i := 0; i < len(x); i++ {
    fmt.Fprintf(w, "%v %v\n", x[i], y[i])
  }
}

/* -------------------------------------------------------------------------- */

func combine(config Config, summary, stability, filename_out string, filename_ins ...string) {
  classifiers := make([]*KmerLrEnsemble, len(filename_ins))
  for i, filename_in := range filename_ins {
    classifiers[i] = ImportKmerLrEnsemble(config, filename_in)
  }
  r := &KmerLrEnsemble{}
  for _, classifier := range classifiers {
    if err := r.AddKmerLrEnsemble(classifier); err != nil {
      log.Fatal(err)
    }
  }
  // stability analysis
  if stability != "" {
    stability_analysis(config, r, stability)
  }
  switch strings.ToLower(summary) {
  case "mean":
    r = r.Mean()
  case "max":
    r = r.Max()
  case "min":
    r = r.Min()
  default:
    log.Fatalf("invalid summary statistic: %s", summary)
  }
  // export model
  SaveModel(config, filename_out, r)
}

/* -------------------------------------------------------------------------- */

func main_combine(config Config, args []string) {
  options := getopt.New()

  optSummary   := options.StringLong("summary",    0 , "mean", "summary [mean (default), max, min]")
  optStability := options.StringLong("stability",  0 ,     "", "export kmer stability analysis")
  optHelp      := options.  BoolLong("help",      'h',         "print help")

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

  combine(config, *optSummary, *optStability, filename_out, filename_ins...)
}
