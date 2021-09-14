/* Copyright (C) 2021 Philipp Benner
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
import   "os"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func expand_scores(config Config, filename_in, filename_out string) {
}

/* -------------------------------------------------------------------------- */

func main_expand_scores(config Config, args []string) {
  options := getopt.New()

  optHelp   := options.BoolLong("help",   'h', "print help")

  options.SetParameters("<SCORES.table> [RESULT.table]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 2 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_in   := options.Args()[1]
  filename_out  := ""
  if len(options.Args()) == 3 {
    filename_out = options.Args()[2]
  }
  expand_scores(config, filename_in, filename_out)
}
