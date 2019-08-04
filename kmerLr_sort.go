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
import   "sort"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type FloatKmer struct {
  a []float64
  b []KmerClass
}

func (obj FloatKmer) Len() int {
  return len(obj.a)
}

func (obj FloatKmer) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj FloatKmer) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

func (obj FloatKmer) Sort() {
  sort.Sort(obj)
}

func (obj FloatKmer) SortReverse() {
  sort.Sort(sort.Reverse(obj))
}

/* -------------------------------------------------------------------------- */

type FloatInt struct {
  a []float64
  b []int
}

func (obj FloatInt) Len() int {
  return len(obj.a)
}

func (obj FloatInt) Swap(i, j int) {
  obj.a[i], obj.a[j] = obj.a[j], obj.a[i]
  obj.b[i], obj.b[j] = obj.b[j], obj.b[i]
}

func (obj FloatInt) Less(i, j int) bool {
  return obj.a[i] < obj.a[j]
}

func (obj FloatInt) Sort() {
  sort.Sort(obj)
}

func (obj FloatInt) SortReverse() {
  sort.Sort(sort.Reverse(obj))
}
