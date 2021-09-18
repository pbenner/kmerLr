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
import   "bufio"
import   "log"
import   "math"
import   "os"
import   "strings"

import . "github.com/pbenner/autodiff"
import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type OperationUnary struct {
  Func   func(float64) float64
  Name   func(string ) string
  Final  bool
}

func (op OperationUnary) apply(columns [][]float64, column_in []float64, names []string, name_in string) ([][]float64, []string) {
  n      := len(columns[0])
  column := make([]float64, n)
  for i := 0; i < n; i++ {
    column[i] = op.Func(column_in[i])
    // check if operation is valid
    if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
      return columns, names
    }
  }
  columns = append(columns, column)
  if len(names) > 0 {
    names = append(names, op.Name(name_in))
  }
  return columns, names
}

func (op OperationUnary) Apply(columns, columns_incomplete [][]float64, names []string, from, to int) ([][]float64, [][]float64, []string) {
  for j := from; j < to; j++ {
    columns, names = op.apply(columns, columns[j], names, names[j])
  }
  return columns, columns_incomplete, names
}

/* -------------------------------------------------------------------------- */

type OperationBinary struct {
  Func   func(float64, float64) float64
  Name   func(string , string ) string
  Final  bool
}

func (op OperationBinary) apply(columns [][]float64, column_a, column_b []float64, names []string, name_a, name_b string) ([][]float64, []string) {
  n      := len(columns[0])
  column := make([]float64, n)
  for i := 0; i < n; i++ {
    column[i] = op.Func(column_a[i], column_b[i])
    // check if operation is valid
    if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
      return columns, names
    }
  }
  columns = append(columns, column)
  if len(names) > 0 {
    names = append(names, op.Name(name_a, name_b))
  }
  return columns, names
}

func (op OperationBinary) Apply(columns, columns_incomplete [][]float64, names []string, from, to int) ([][]float64, [][]float64, []string) {
  for j1 := from; j1 < to; j1++ {
    for j2 := j1+1; j2 < to; j2++ {
      columns, names = op.apply(columns, columns[j1], columns[j2], names, names[j1], names[j2])
    }
  }
  return columns, columns_incomplete, names
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
  Name : func(a string ) string  { return fmt.Sprintf("(%s^2)", a) },
  Final: true }

var _sqrt OperationUnary = OperationUnary{
  Func : func(a float64) float64 { return math.Sqrt(a) },
  Name : func(a string ) string  { return fmt.Sprintf("sqrt(%s)", a) },
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

func expand_import(config Config, filenames_in []string) ([][]float64, []int, []string) {
  scores, names := compile_data_scores(config, nil, nil, nil, true, filenames_in...)
  // merge scores for easier looping
  scores_merged  := []DenseFloat64Vector{}
  lengths := []int{}
  for i := 0; i < len(scores); i++ {
    lengths = append(lengths, len(scores[i]))
    for j := 0; j < len(scores[i]); j++ {
      scores_merged = append(scores_merged, AsDenseFloat64Vector(scores[i][j]))
    }
  }
  // transpose data
  columns := [][]float64{}
  for j := 1; j < len(scores_merged[0]); j++ {
    t := make([]float64, len(scores_merged))
    for i := 0; i < len(scores_merged); i++ {
      t[i] = scores_merged[i].Float64At(j)
    }
    columns = append(columns, t)
  }
  if len(columns) == 0 {
    log.Fatal("No data given. Exiting.")
    panic("internal error")
  }
  return columns, lengths, names
}

func expand_export(config Config, columns [][]float64, lengths []int, names []string, basename_out string) {
  offset := 0
  for i_ := 0; i_ < len(lengths); i_++ {
    f, err := os.Create(fmt.Sprintf("%s_%d.table", basename_out, i_))
    if err != nil {
      panic(err)
    }
    defer f.Close()

    w := bufio.NewWriter(f)
    defer w.Flush()

    // print header
    if len(names) > 0 {
      for j, name := range names {
        if j == 0 {
          fmt.Fprintf(w,  "%s", name)
        } else {
          fmt.Fprintf(w, ",%s", name)
        }
      }
      fmt.Fprintf(w, "\n")
    }
    // print data
    for i := offset; i < offset+lengths[i_]; i++ {
      for j := 0; j < len(columns); j++ {
        if j == 0 {
          fmt.Fprintf(w,  "%e", columns[j][i])
        } else {
          fmt.Fprintf(w, ",%e", columns[j][i])
        }
      }
      fmt.Fprintf(w, "\n")
    }
    offset += lengths[i_]
  }
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

  columns, lengths, names := expand_import(config, filenames_in)
  columns_incomplete      := [][]float64{}
  
  from := 0
  to   := len(columns)
  for d_ := 0; d_ < d; d_++ {
    // apply unary operations
    for _, op := range op_unary {
      columns, columns_incomplete, names = op.Apply(columns, columns_incomplete, names, from, to)
    }
    // apply binary operations
    for _, op := range op_binary {
      columns, columns_incomplete, names = op.Apply(columns, columns_incomplete, names, from, to)
    }
    // update range
    from = to
    to   = len(columns)
  }
  expand_export(config, columns, lengths, names, basename_out)
}

/* -------------------------------------------------------------------------- */

func main_expand_scores(config Config, args []string) {
  options := getopt.New()

  optDepth  := options. IntLong("depth",   0 , 1, "recursion depth")
  optHeader := options.BoolLong("header",  0 ,    "input files contain a header with feature names")
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
  config.Header = *optHeader
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 2 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filenames_in := strings.Split(options.Args()[0], ",")
  basename_out := options.Args()[1]

  expand_scores(config, filenames_in, basename_out, *optDepth)
}
