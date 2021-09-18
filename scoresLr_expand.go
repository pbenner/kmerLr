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

type AbstractOperation interface {
  Apply(columns, incomplete_columns [][]float64, names, incomplete_names []string, from, to, max_features int) ([][]float64, [][]float64, []string, []string)
}

/* -------------------------------------------------------------------------- */

type Operation struct {
  Final bool
}

func (op Operation) Append(columns, incomplete_columns [][]float64, names, incomplete_names []string, column []float64, name string) ([][]float64, [][]float64, []string, []string) {
  if column != nil {
    if op.Final {
      columns = append(columns, column)
      if len(names) > 0 {
        names = append(names, name)
      }
    } else {
      incomplete_columns = append(incomplete_columns, column)
      if len(names) > 0 {
        incomplete_names = append(incomplete_names, name)
      }
    }
  }
  return columns, incomplete_columns, names, incomplete_names
}

/* -------------------------------------------------------------------------- */

type OperationUnary struct {
  Operation
  Func func(float64) float64
  Name func(string ) string
}

func (op OperationUnary) apply(column_in []float64, name_in string) ([]float64, string) {
  column := make([]float64, len(column_in))
  name   := ""
  for i := 0; i < len(column_in); i++ {
    column[i] = op.Func(column_in[i])
    // check if operation is valid
    if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
      return nil, name
    }
  }
  if name_in != "" {
    name = op.Name(name_in)
  }
  return column, name
}

func (op OperationUnary) Apply(columns, incomplete_columns [][]float64, names, incomplete_names []string, from, to, max_features int) ([][]float64, [][]float64, []string, []string) {
  tmp_columns := columns[from:to]
  tmp_names   :=   names[from:to]
  if op.Final {
    tmp_columns = append(tmp_columns, incomplete_columns...)
    tmp_names   = append(tmp_names  , incomplete_names  ...)
    incomplete_columns = nil
    incomplete_names   = nil
  }
  // apply to complete
  for j := 0; j < len(tmp_columns); j++ {
    if max_features > 0 && len(columns) >= max_features {
      break
    }
    column, name := op.apply(tmp_columns[j], tmp_names[j])
    columns, incomplete_columns, names, incomplete_names = op.Append(columns, incomplete_columns, names, incomplete_names, column, name)
  }
  return columns, incomplete_columns, names, incomplete_names
}

/* -------------------------------------------------------------------------- */

type OperationBinary struct {
  Operation
  Func func(float64, float64) float64
  Name func(string , string ) string
}

func (op OperationBinary) apply(column_a, column_b []float64, name_a, name_b string) ([]float64, string) {
  n      := len(column_a)
  column := make([]float64, n)
  name   := ""
  for i := 0; i < n; i++ {
    column[i] = op.Func(column_a[i], column_b[i])
    // check if operation is valid
    if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
      return nil, name
    }
  }
  if name_a != "" {
    name = op.Name(name_a, name_b)
  }
  return column, name
}

func (op OperationBinary) Apply(columns, incomplete_columns [][]float64, names, incomplete_names []string, from, to, max_features int) ([][]float64, [][]float64, []string, []string) {
  tmp_columns := columns[from:to]
  tmp_names   :=   names[from:to]
  if op.Final {
    tmp_columns = append(tmp_columns, incomplete_columns...)
    tmp_names   = append(tmp_names  , incomplete_names  ...)
    incomplete_columns = nil
    incomplete_names   = nil
  }
  for j1 := 0; j1 < len(tmp_columns); j1++ {
    for j2 := j1+1; j2 < len(tmp_columns); j2++ {
      if max_features > 0 && len(columns) >= max_features {
        goto ret
      }
      column, name := op.apply(tmp_columns[j1], tmp_columns[j2], tmp_names[j1], tmp_names[j2])
      columns, incomplete_columns, names, incomplete_names = op.Append(columns, incomplete_columns, names, incomplete_names, column, name)
    }
  }
ret:
  return columns, incomplete_columns, names, incomplete_names
}

/* -------------------------------------------------------------------------- */

var _exp OperationUnary = OperationUnary{
  Operation: Operation{true},
  Func : func(a float64) float64 { return math.Exp(a) },
  Name : func(a string ) string  { return fmt.Sprintf("exp(%s)", a) } }

var _log OperationUnary = OperationUnary{
  Operation: Operation{true},
  Func : func(a float64) float64 { return math.Log(a) },
  Name : func(a string ) string  { return fmt.Sprintf("log(%s)", a) } }

var _square OperationUnary = OperationUnary{
  Operation: Operation{true},
  Func : func(a float64) float64 { return a*a },
  Name : func(a string ) string  { return fmt.Sprintf("(%s^2)", a) } }

var _sqrt OperationUnary = OperationUnary{
  Operation: Operation{true},
  Func : func(a float64) float64 { return math.Sqrt(a) },
  Name : func(a string ) string  { return fmt.Sprintf("sqrt(%s)", a) } }

var _add OperationBinary = OperationBinary{
  Operation: Operation{false},
  Func : func(a, b float64) float64 { return a+b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s+%s)", a, b) } }

var _sub OperationBinary = OperationBinary{
  Operation: Operation{false},
  Func : func(a, b float64) float64 { return a-b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s-%s)", a, b) } }

var _subrev OperationBinary = OperationBinary{
  Operation: Operation{false},
  Func : func(a, b float64) float64 { return b-a },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s-%s)", b, a) } }

var _mul OperationBinary = OperationBinary{
  Operation: Operation{true},
  Func : func(a, b float64) float64 { return a*b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s*%s)", a, b) } }

var _div OperationBinary = OperationBinary{
  Operation: Operation{true},
  Func : func(a, b float64) float64 { return a/b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s/%s)", a, b) } }

var _divrev OperationBinary = OperationBinary{
  Operation: Operation{true},
  Func : func(a, b float64) float64 { return b/a },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s/%s)", b, a) } }

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

func expand_scores(config Config, filenames_in []string, basename_out string, max_d, max_features int, allowed_operations string) {
  operations := []AbstractOperation{}
  operations  = append(operations, _exp)
  operations  = append(operations, _log)
  operations  = append(operations, _square)
  operations  = append(operations, _sqrt)
  operations  = append(operations, _add)
  operations  = append(operations, _sub)
  operations  = append(operations, _subrev)
  operations  = append(operations, _mul)
  operations  = append(operations, _div)
  operations  = append(operations, _divrev)

  columns, lengths, names := expand_import(config, filenames_in)
  incomplete_columns := [][]float64{}
  incomplete_names   := []string{}

  from := 0
  to   := len(columns)
  for d := 0; max_d == 0 || d < max_d; d++ {
    for _, op := range operations {
      columns, incomplete_columns, names, incomplete_names = op.Apply(columns, incomplete_columns, names, incomplete_names, from, to, max_features)
    }
    if max_features > 0 && len(columns) >= max_features {
      break
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

  optMaxDepth    := options.   IntLong("max-depth",     0 ,   0, "maximum recursion depth, zero for no maximum [default: 0]")
  optMaxFeatures := options.   IntLong("max-features",  0 , 100, "maximum number of features, zero for no maximum [default: 0]")
  optOperations  := options.StringLong("operations",    0 ,  "", "list of allowed operations")
  optHeader      := options.  BoolLong("header",        0 ,      "input files contain a header with feature names")
  optHelp        := options.  BoolLong("help",         'h',      "print help")

  options.SetParameters("<SCORES1.table,SCORES2.table,...> <BASENAME_OUT>")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optMaxDepth < 0 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optMaxFeatures < 0 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optMaxDepth == 0 && *optMaxFeatures == 0 {
    log.Fatal("Recursion depth or number of features must be constrained")
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

  expand_scores(config, filenames_in, basename_out, *optMaxDepth, *optMaxFeatures, *optOperations)
}
