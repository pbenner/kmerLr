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
import   "strconv"
import   "strings"

import . "github.com/pbenner/autodiff"
import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type AbstractOperation interface {
  Apply(desk Desk, from, to, max_features int, max_value float64) Desk
}

/* -------------------------------------------------------------------------- */

type Operation struct {
  Final           bool
  Incompatible []*Operation
}

func (op *Operation) AddIncompatible(op_a *Operation) {
  op.Incompatible = append(op.Incompatible, op_a)
}

func (op *Operation) IsIncompatible(op_a *Operation) bool {
  if op == nil {
    return false
  }
  for _, op_b := range op.Incompatible {
    if op_a == op_b {
      return true
    }
  }
  return false
}

/* -------------------------------------------------------------------------- */

type Desk struct {
  columns            [][]float64
  names                []string
  lastop               []*Operation
  incomplete_columns [][]float64
  incomplete_names     []string
  incomplete_lastop    []*Operation
}

func (obj Desk) GetColumns(from, to int) [][]float64 {
  return obj.columns[from:to]
}

func (obj Desk) GetLastop(from, to int) []*Operation {
  return obj.lastop[from:to]
}

func (obj Desk) GetNames(from, to int) []string {
  if len(obj.names) == 0 {
    return nil
  } else {
    return obj.names[from:to]
  }
}

func (obj Desk) Append(column []float64, name string, op *Operation) Desk {
  if column != nil {
    if op.Final {
      obj.columns = append(obj.columns, column)
      obj.lastop  = append(obj.lastop , op)
      if len(obj.names) > 0 {
        obj.names = append(obj.names, name)
      }
    } else {
      obj.incomplete_columns = append(obj.incomplete_columns, column)
      obj.incomplete_lastop  = append(obj.incomplete_lastop , op)
      if len(obj.names) > 0 {
        obj.incomplete_names = append(obj.incomplete_names, name)
      }
    }
  }
  return obj
}

/* -------------------------------------------------------------------------- */

type OperationUnary struct {
  *Operation
  Func func(float64) float64
  Name func(string ) string
}

func (op OperationUnary) apply(column_in []float64, name_in string, max_value float64) ([]float64, string) {
  column := make([]float64, len(column_in))
  name   := ""
  for i := 0; i < len(column_in); i++ {
    column[i] = op.Func(column_in[i])
    // check if operation is valid
    if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
      return nil, name
    }
    if max_value > 0.0 && math.Abs(column[i]) > max_value {
      return nil, name
    }
  }
  if name_in != "" {
    name = op.Name(name_in)
  }
  return column, name
}

func (op OperationUnary) Apply(desk Desk, from, to, max_features int, max_value float64) Desk {
  tmp_columns := desk.GetColumns(from, to)
  tmp_lastop  := desk.GetLastop (from, to)
  tmp_names   := desk.GetNames  (from, to)
  if op.Final {
    tmp_columns = append(tmp_columns, desk.incomplete_columns...)
    tmp_lastop  = append(tmp_lastop , desk.incomplete_lastop ...)
    tmp_names   = append(tmp_names  , desk.incomplete_names  ...)
    desk.incomplete_columns = nil
    desk.incomplete_lastop  = nil
    desk.incomplete_names   = nil
  }
  // apply to complete
  for j := 0; j < len(tmp_columns); j++ {
    if op.IsIncompatible(tmp_lastop[j]) {
      continue
    }
    if max_features > 0 && len(desk.columns) >= max_features {
      break
    }
    tmp_name := ""
    if len(tmp_names) > 0 {
      tmp_name = tmp_names[j]
    }
    column, name := op.apply(tmp_columns[j], tmp_name, max_value)
    desk = desk.Append(column, name, op.Operation)
  }
  return desk
}

/* -------------------------------------------------------------------------- */

type OperationBinary struct {
  *Operation
  Func func(float64, float64) float64
  Name func(string , string ) string
}

func (op OperationBinary) apply(column_a, column_b []float64, name_a, name_b string, max_value float64) ([]float64, string) {
  n      := len(column_a)
  column := make([]float64, n)
  name   := ""
  for i := 0; i < n; i++ {
    column[i] = op.Func(column_a[i], column_b[i])
    // check if operation is valid
    if math.IsNaN(column[i]) || math.IsInf(column[i], 0) {
      return nil, name
    }
    if max_value > 0.0 && math.Abs(column[i]) > max_value {
      return nil, name
    }
  }
  if name_a != "" {
    name = op.Name(name_a, name_b)
  }
  return column, name
}

func (op OperationBinary) Apply(desk Desk, from, to, max_features int, max_value float64) Desk {
  tmp_columns := desk.GetColumns(from, to)
  tmp_lastop  := desk.GetLastop (from, to)
  tmp_names   := desk.GetNames  (from, to)
  if op.Final {
    tmp_columns = append(tmp_columns, desk.incomplete_columns...)
    tmp_lastop  = append(tmp_lastop , desk.incomplete_lastop ...)
    tmp_names   = append(tmp_names  , desk.incomplete_names  ...)
    desk.incomplete_columns = nil
    desk.incomplete_lastop  = nil
    desk.incomplete_names   = nil
  }
  for j1 := 0; j1 < len(tmp_columns); j1++ {
    if op.IsIncompatible(tmp_lastop[j1]) {
      continue
    }
    for j2 := j1+1; j2 < len(tmp_columns); j2++ {
      if op.IsIncompatible(tmp_lastop[j2]) {
        continue
      }
      if max_features > 0 && len(desk.columns) >= max_features {
        goto ret
      }
      tmp_name_a := ""
      tmp_name_b := ""
      if len(tmp_names) > 0 {
        tmp_name_a = tmp_names[j1]
        tmp_name_b = tmp_names[j2]
      }
      column, name := op.apply(tmp_columns[j1], tmp_columns[j2], tmp_name_a, tmp_name_b, max_value)
      desk = desk.Append(column, name, op.Operation)
    }
  }
ret:
  return desk
}

/* -------------------------------------------------------------------------- */

var _exp OperationUnary = OperationUnary{
  Operation: &Operation{Final: true},
  Func : func(a float64) float64 { return math.Exp(a) },
  Name : func(a string ) string  { return fmt.Sprintf("exp(%s)", a) } }

var _expneg OperationUnary = OperationUnary{
  Operation: &Operation{Final: true},
  Func : func(a float64) float64 { return math.Exp(-a) },
  Name : func(a string ) string  { return fmt.Sprintf("exp(-%s)", a) } }

var _log OperationUnary = OperationUnary{
  Operation: &Operation{Final: true},
  Func : func(a float64) float64 { return math.Log(a) },
  Name : func(a string ) string  { return fmt.Sprintf("log(%s)", a) } }

var _square OperationUnary = OperationUnary{
  Operation: &Operation{Final: true},
  Func : func(a float64) float64 { return a*a },
  Name : func(a string ) string  { return fmt.Sprintf("(%s^2)", a) } }

var _sqrt OperationUnary = OperationUnary{
  Operation: &Operation{Final: true},
  Func : func(a float64) float64 { return math.Sqrt(a) },
  Name : func(a string ) string  { return fmt.Sprintf("sqrt(%s)", a) } }

var _add OperationBinary = OperationBinary{
  Operation: &Operation{Final: false},
  Func : func(a, b float64) float64 { return a+b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s+%s)", a, b) } }

var _sub OperationBinary = OperationBinary{
  Operation: &Operation{Final: false},
  Func : func(a, b float64) float64 { return a-b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s-%s)", a, b) } }

var _subrev OperationBinary = OperationBinary{
  Operation: &Operation{Final: false},
  Func : func(a, b float64) float64 { return b-a },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s-%s)", b, a) } }

var _mul OperationBinary = OperationBinary{
  Operation: &Operation{Final: true},
  Func : func(a, b float64) float64 { return a*b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s*%s)", a, b) } }

var _div OperationBinary = OperationBinary{
  Operation: &Operation{Final: true},
  Func : func(a, b float64) float64 { return a/b },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s/%s)", a, b) } }

var _divrev OperationBinary = OperationBinary{
  Operation: &Operation{Final: true},
  Func : func(a, b float64) float64 { return b/a },
  Name : func(a, b string ) string  { return fmt.Sprintf("(%s/%s)", b, a) } }

/* -------------------------------------------------------------------------- */

func init() {
  _exp.AddIncompatible(_exp.Operation)
  _exp.AddIncompatible(_log.Operation)
  _exp.AddIncompatible(_expneg.Operation)
  _log.AddIncompatible(_exp.Operation)
  _log.AddIncompatible(_log.Operation)
  _log.AddIncompatible(_expneg.Operation)
  _expneg.AddIncompatible(_exp.Operation)
  _expneg.AddIncompatible(_log.Operation)
  _expneg.AddIncompatible(_expneg.Operation)
}

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

func expand_parse_operations(allowed_operations string) []AbstractOperation {
  operations := []AbstractOperation{}
  if allowed_operations == "" {
    operations = append(operations, _exp)
    operations = append(operations, _expneg)
    operations = append(operations, _log)
    operations = append(operations, _square)
    operations = append(operations, _sqrt)
    operations = append(operations, _add)
    operations = append(operations, _sub)
    operations = append(operations, _subrev)
    operations = append(operations, _mul)
    operations = append(operations, _div)
    operations = append(operations, _divrev)
  } else {
    for _, op_str := range strings.Split(allowed_operations, ",") {
      switch op_str {
      case "exp":
        operations = append(operations, _exp)
        operations = append(operations, _expneg)
      case "log":
        operations = append(operations, _log)
      case "square":
        operations = append(operations, _square)
      case "sqrt":
        operations = append(operations, _sqrt)
      case "add":
        operations = append(operations, _add)
      case "sub":
        operations = append(operations, _sub)
        operations = append(operations, _subrev)
      case "mul":
        operations = append(operations, _mul)
      case "div":
        operations = append(operations, _div)
        operations = append(operations, _divrev)
      default:
        log.Fatalf("Invalid operation `%s'. Exiting.", op_str)
      }
    }
  }
  return operations
}

/* -------------------------------------------------------------------------- */

func expand_scores(config Config, filenames_in []string, basename_out string, max_d, max_features int, max_value float64, operations []AbstractOperation) {
  desk    := Desk{}
  lengths := []int{}
  // import data
  if columns, lengths_, names := expand_import(config, filenames_in); true {
    desk.columns = columns
    desk.names   = names
    desk.lastop  = make([]*Operation, len(columns))
    lengths      = lengths_
  }
  from := 0
  to   := len(desk.columns)
  for d := 0; max_d == 0 || d < max_d; d++ {
    for _, op := range operations {
      desk = op.Apply(desk, from, to, max_features, max_value)
    }
    if max_features > 0 && len(desk.columns) >= max_features {
      break
    }
    // update range
    from = to
    to   = len(desk.columns)
  }
  expand_export(config, desk.columns, lengths, desk.names, basename_out)
}

/* -------------------------------------------------------------------------- */

func main_expand_scores(config Config, args []string) {
  options := getopt.New()

  optMaxDepth    := options.   IntLong("max-depth",     0 ,   0, "maximum recursion depth, zero for no maximum [default: 0]")
  optMaxFeatures := options.   IntLong("max-features",  0 , 100, "maximum number of features, zero for no maximum [default: 0]")
  optMaxValue    := options.StringLong("max-value",     0 , "0", "maximum absolute value")
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
  max_value := 0.0
  if v, err := strconv.ParseFloat(*optMaxValue, 64); err != nil {
    log.Fatalf("Parsing option `--max-value' failed: %v", err)
  } else {
    max_value = v
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

  expand_scores(config, filenames_in, basename_out, *optMaxDepth, *optMaxFeatures, max_value, expand_parse_operations(*optOperations))
}
