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
import   "strings"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func alphabet_from_string(str string) (ComplementableAlphabet, error) {
  str = strings.ToLower(str)
  str = strings.Replace(str, "-", " ", -1)
  if !strings.HasSuffix(str, "alphabet") {
    str += " alphabet"
  }
  switch str {
  case (NucleotideAlphabet{}).String():
    return NucleotideAlphabet{}, nil
  case (GappedNucleotideAlphabet{}).String():
    return GappedNucleotideAlphabet{}, nil
  case (AmbiguousNucleotideAlphabet{}).String():
    return AmbiguousNucleotideAlphabet{}, nil
  default:
    return nil, fmt.Errorf("invalid alphabet")
  }
}
