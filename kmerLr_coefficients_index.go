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
import   "math"

/* -------------------------------------------------------------------------- */

type CoeffIndex int

/* -------------------------------------------------------------------------- */

func (obj CoeffIndex) Ind2Sub(k1, k2 int) int {
  n := int(obj)
  if k1 == k2 {
    return k1+1
  } else {
    return n + (n*(n-1)/2) - (n-k1)*((n-k1)-1)/2 + k2 - k1
  }
}

func (obj CoeffIndex) Sub2Ind(i int) (int, int) {
  n := int(obj)
  if i < n {
    return i, i
  } else {
    i   = i - n
    k1 := n - 2 - int(math.Floor(math.Sqrt(float64(-8*i + 4*n*(n-1)-7))/2.0 - 0.5))
    k2 := i + k1 + 1 - n*(n-1)/2 + (n-k1)*((n-k1)-1)/2
    return k1, k2
  }
}
