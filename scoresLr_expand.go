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

import   "fmt"
import   "math"
import   "os"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type f_unary  func(float64         ) float64
type f_binary func(float64, float64) float64

type OperationUnary struct {
  Func   f_unary
  Name   func(string) string
  Final  bool
}

type OperationBinary struct {
  Func   f_binary
  Name   func(string, string) string
  Final  bool
}

/* -------------------------------------------------------------------------- */


var _exp OperationUnary = OperationUnary{
  Func : func(a float64) float64 { return math.Exp(a) },
  Name : func(a string ) string  { return fmt.Sprintf("exp(%s)", a) },
  Final: true }

var _log OperationUnary = OperationUnary{
  Func : func(a float64) float64 { return math.Log(a) },
  Name : func(a string ) string  { return fmt.Sprintf("log(%s)", a) },
  Final: true }

var _add OperationBinary = OperationBinary{
  Func : func(a, b float64) float64 { return a+b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s+%s)", a, b) },
  Final: false }

var _sub OperationBinary = OperationBinary{
  Func : func(a, b float64) float64 { return a-b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s-%s)", a, b) },
  Final: false }

var _subrev OperationBinary = OperationBinary{
  Func : func(a, b float64) float64 { return b-a },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s-%s)", b, a) },
  Final: false }

var _mul OperationBinary = OperationBinary{
  Func : func(a, b float64) float64 { return a*b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s*%s)", a, b) },
  Final: true }

var _div OperationBinary = OperationBinary{
  Func : func(a, b float64) float64 { return a/b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s/%s)", a, b) },
  Final: true }

var _divrev OperationBinary = OperationBinary{
  Func : func(a, b float64) float64 { return b/a },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s/%s)", b, a) },
  Final: true }

/* -------------------------------------------------------------------------- */

func expand_scores(config Config, filename_in, filename_out string, d int) {
  op_unary := []OperationUnary{}
  op_unary  = append(op_unary, _exp)
  op_unary  = append(op_unary, _log)
  op_binary := []OperationBinary{}
  op_binary  = append(op_binary, _add)
  op_binary  = append(op_binary, _sub)
  op_binary  = append(op_binary, _subrev)
  op_binary  = append(op_binary, _mul)
  op_binary  = append(op_binary, _div)
  op_binary  = append(op_binary, _divrev)
}

/* -------------------------------------------------------------------------- */

func main_expand_scores(config Config, args []string) {
  options := getopt.New()

  optDepth  := options. IntLong("depth",   0 , 2, "recursion depth")
  optHelp   := options.BoolLong("help",   'h',    "print help")

  options.SetParameters("<SCORES.table> [RESULT.table]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optDepth < 0 {
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
  expand_scores(config, filename_in, filename_out, *optDepth)
}
