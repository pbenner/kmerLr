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
import   "log"
import   "math"
import   "os"
import   "strings"

import . "github.com/pbenner/autodiff"
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

var _square OperationUnary = OperationUnary{
  Func : func(a float64) float64 { return a*a },
  Name : func(a string ) string  { return fmt.Sprintf("(a^2)", a) },
  Final: true }

var _sqrt OperationUnary = OperationUnary{
  Func : func(a float64) float64 { return math.Sqrt(a) },
  Name : func(a string ) string  { return fmt.Sprintf("sqrt(a)", a) },
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

func apply_unary(config Config, columns [][]float64, names []string, op OperationUnary, from, to int) ([][]float64, []string) {
  n      := len(columns[0])
  column := make([]float64, n)
  for j := from; j < to; j++ {
    for i := 0; i < n; i++ {
      column[i] = op.Func(columns[j][i])
      // check if operation is valid
      if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
        break
      }
      // append new column
      columns = append(columns, column)
      // generate new column name
      if len(names) > 0 {
        names = append(names, op.Name(names[j]))
      }
    }
  }
  return columns, names
}

/* -------------------------------------------------------------------------- */

func expand_import(config Config, filenames_in []string) ([][]float64, []int, []string) {
  scores, names := compile_data_scores(config, nil, nil, nil, true, filenames_in...)
  // merge scores for easier looping
  scores_merged  := []DenseFloat64Vector{}
  scores_lengths := []int{}
  for i := 0; i < len(scores); i++ {
    scores_lengths = append(scores_lengths, len(scores[i]))
    for j := 0; j < len(scores[i]); j++ {
      scores_merged = append(scores_merged, AsDenseFloat64Vector(scores[i][j]))
    }
  }
  // transpose data
  scores_columns := [][]float64{}
  for j := 1; j < len(scores_merged[0]); j++ {
    t := make([]float64, len(scores_merged[0]))
    for i := 0; i < len(scores_merged); i++ {
      t[i] = scores_merged[i].Float64At(j)
    }
    scores_columns = append(scores_columns, t)
  }
  if len(scores_columns) == 0 {
    log.Fatal("No data given. Exiting.")
    panic("internal error")
  }
  return scores_columns, scores_lengths, names
}

/* -------------------------------------------------------------------------- */

func expand_scores(config Config, filenames_in []string, basename_out string, d int) {
  op_unary := []OperationUnary{}
  op_unary  = append(op_unary, _exp)
  op_unary  = append(op_unary, _log)
  op_unary  = append(op_unary, _square)
  op_unary  = append(op_unary, _sqrt)
  op_binary := []OperationBinary{}
  op_binary  = append(op_binary, _add)
  op_binary  = append(op_binary, _sub)
  op_binary  = append(op_binary, _subrev)
  op_binary  = append(op_binary, _mul)
  op_binary  = append(op_binary, _div)
  op_binary  = append(op_binary, _divrev)

  scores_columns, _, names := expand_import(config, filenames_in)
  
  from := 0
  to   := len(scores_columns)
  for d_ := 0; d_ < d; d++ {
    // apply unary operations
    for _, op := range op_unary {
      scores_columns, names = apply_unary(config, scores_columns, names, op, from, to)
    }
    // apply binary operations
    // for _, op := range op_binary {
    //   names = apply_binary(config, scores_columns, names, op, from, to)
    // }
    // update range
    from = to
    to   = len(scores_columns)
  }
}

/* -------------------------------------------------------------------------- */

func main_expand_scores(config Config, args []string) {
  options := getopt.New()

  optDepth  := options. IntLong("depth",   0 , 2, "recursion depth")
  optHelp   := options.BoolLong("help",   'h',    "print help")

  options.SetParameters("<SCORES1.table,SCORES2.table,...> <BASENAME_OUT>")
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
  filenames_in := strings.Split(options.Args()[1], ",")
  basename_out := options.Args()[2]

  expand_scores(config, filenames_in, basename_out, *optDepth)
}
