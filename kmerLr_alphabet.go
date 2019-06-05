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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type AlphabetDef struct {
  M, N           int
  Binarize       bool
  Complement     bool
  Reverse        bool
  Revcomp        bool
  MaxAmbiguous []int
  Alphabet       ComplementableAlphabet
}

/* -------------------------------------------------------------------------- */

func (obj *AlphabetDef) ImportConfig(config ConfigDistribution, t ScalarType) error {
  m, ok := config.GetNamedParameterAsInt("M"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  n, ok := config.GetNamedParameterAsInt("N"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  binarize, ok := config.GetNamedParameterAsBool("Binarize"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  complement, ok := config.GetNamedParameterAsBool("Complement"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  reverse, ok := config.GetNamedParameterAsBool("Reverse"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  revcomp, ok := config.GetNamedParameterAsBool("Revcomp"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  maxAmbiguous, ok := config.GetNamedParametersAsInts("MaxAmbiguous"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  alphabet, ok := config.GetNamedParameterAsString("Alphabet"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  if r, err := alphabet_from_string(alphabet); err != nil {
    return err
  } else {
    obj.Alphabet = r
  }
  obj.M, obj.N     = m, n
  obj.Binarize     = binarize
  obj.Complement   = complement
  obj.Reverse      = reverse
  obj.Revcomp      = revcomp
  obj.MaxAmbiguous = maxAmbiguous
  return nil
}

func (obj *AlphabetDef) ExportConfig() ConfigDistribution {
  config := struct{
    M, N           int
    Binarize       bool
    Complement     bool
    Reverse        bool
    Revcomp        bool
    MaxAmbiguous []int
    Alphabet       string
  }{}
  config.M, config.N  = obj.M, obj.N
  config.Binarize     = obj.Binarize
  config.Complement   = obj.Complement
  config.Reverse      = obj.Reverse
  config.Revcomp      = obj.Revcomp
  config.MaxAmbiguous = obj.MaxAmbiguous
  config.Alphabet     = obj.Alphabet.String()

  return NewConfigDistribution("alphabet", config)
}

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
