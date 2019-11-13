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
import   "math"
import   "os"
import   "time"

/* -------------------------------------------------------------------------- */

func format_duration(duration time.Duration) string {
  days    := int(duration.Hours() / 24)
	hours   := int(math.Mod(duration.Hours(), 24))
	minutes := int(math.Mod(duration.Minutes(), 60))
	seconds := int(math.Mod(duration.Seconds(), 60))
	msecs   := int(math.Mod(float64(duration.Milliseconds()), 1000))
  return fmt.Sprintf("%02d:%02d:%02d:%02d.%03d", days, hours, minutes, seconds, msecs)
}

/* -------------------------------------------------------------------------- */

type Trace struct {
  Iteration []int
  Nonzero   []int
  Change    []float64
  Lambda    []float64
  Loss      []float64
  Duration  []time.Duration
}

func (obj Trace) Export(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  // print header
  fmt.Fprintf(w, "%15s %6s %12s %8s", "duration", "iteration", "change", "nonzero")
  if len(obj.Lambda) > 0 {
    fmt.Fprintf(w, " %12s", "lambda")
  }
  if len(obj.Loss) > 0 {
    fmt.Fprintf(w, " %12s", "loss")
  }
  fmt.Fprintf(w, "\n")

  // print values
  for i := 0; i < obj.Length(); i++ {
    fmt.Fprintf(w, "%15v %6d %12e %8d", format_duration(obj.Duration[i]), obj.Iteration[i], obj.Change[i], obj.Nonzero[i])
    if len(obj.Lambda) > 0 {
      fmt.Fprintf(w, " %12e", obj.Lambda[i])
    }
    if len(obj.Loss) > 0 {
      fmt.Fprintf(w, " %12e", obj.Loss[i])
    }
    fmt.Fprintf(w, "\n")
  }
  return nil
}

func (obj Trace) Length() int {
  return len(obj.Iteration)
}

func (obj *Trace) Append(iteration, nonzero int, change, lambda, loss float64, duration time.Duration) {
  obj.Iteration = append(obj.Iteration, iteration)
  obj.Nonzero   = append(obj.Nonzero  , nonzero)
  obj.Change    = append(obj.Change   , change)
  if !math.IsNaN(lambda) {
    obj.Lambda = append(obj.Lambda, lambda)
  }
  if !math.IsNaN(loss) {
    obj.Loss = append(obj.Loss, loss)
  }
  obj.Duration    = append(obj.Duration, duration)
}

func (obj Trace) CompVar(n int) float64 {
  if len(obj.Nonzero) < n {
    return math.NaN()
  }
  v := obj.Nonzero[len(obj.Nonzero)-n:len(obj.Nonzero)]
  m := 0.0
  for i := 0; i < n; i++ {
    m += float64(v[i])
  }
  m /= float64(n)
  r := 0.0
  for i := 0; i < n; i++ {
    s := float64(v[i]) - m
    r += s*s
  }
  r /= float64(n)
  return r
}
