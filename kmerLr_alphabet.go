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

type KmerLrAlphabetDef struct {
  M, N           int
  Binarize       bool
  Complement     bool
  Reverse        bool
  Revcomp        bool
  MaxAmbiguous []int
  Alphabet       ComplementableAlphabet
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrAlphabetDef) Clone() KmerLrAlphabetDef {
  r := KmerLrAlphabetDef{}
  r  = obj
  r.MaxAmbiguous = make([]int, len(obj.MaxAmbiguous))
  copy(r.MaxAmbiguous, obj.MaxAmbiguous)
  return r
}

func (a KmerLrAlphabetDef) Equals(b KmerLrAlphabetDef) bool {
  if a.M != b.M {
    return false
  }
  if a.N != b.N {
    return false
  }
  if a.Binarize != b.Binarize {
    return false
  }
  if a.Complement != b.Complement {
    return false
  }
  if a.Reverse != b.Reverse {
    return false
  }
  if a.Revcomp != b.Revcomp {
    return false
  }
  if a.Alphabet.String() != b.Alphabet.String() {
    return false
  }
  if len(a.MaxAmbiguous) != len(b.MaxAmbiguous) {
    return false
  }
  for i := 0; i < len(a.MaxAmbiguous); i++ {
    if a.MaxAmbiguous[i] != b.MaxAmbiguous[i] {
      return false
    }
  }
  return true
}

/* -------------------------------------------------------------------------- */

type KmerLrAlphabet struct {
  KmerLrAlphabetDef
  Kmers KmerClassList
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrAlphabet) Clone() KmerLrAlphabet {
  r := KmerLrAlphabet{}
  r.KmerLrAlphabetDef = obj.KmerLrAlphabetDef.Clone()
  r.Kmers             = obj.Kmers            .Clone()
  return r
}

/* -------------------------------------------------------------------------- */

func (a KmerLrAlphabet) Equals(b KmerLrAlphabet) bool {
  if !a.KmerLrAlphabetDef.Equals(b.KmerLrAlphabetDef) {
    return false
  }
  return a.Kmers.Equals(b.Kmers)
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrAlphabet) ImportConfig(config ConfigDistribution, t ScalarType) error {
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
  kmers, ok := config.GetNamedParametersAsStrings("Kmers"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  if r, err := alphabet_from_string(alphabet); err != nil {
    return err
  } else {
    obj.Alphabet = r
  }
  if rel, err := NewKmerEquivalenceRelation(n, m, complement, reverse, revcomp, maxAmbiguous, obj.Alphabet); err != nil {
    return err
  } else {
    obj.Kmers = make(KmerClassList, len(kmers))
    for i, str := range kmers {
      obj.Kmers[i] = rel.EquivalenceClass(strings.Split(str, "|")[0])
    }
  }
  obj.M, obj.N     = m, n
  obj.Binarize     = binarize
  obj.Complement   = complement
  obj.Reverse      = reverse
  obj.Revcomp      = revcomp
  obj.MaxAmbiguous = maxAmbiguous
  return nil
}

func (obj *KmerLrAlphabet) ExportConfig() ConfigDistribution {
  config := struct{
    M, N           int
    Binarize       bool
    Complement     bool
    Reverse        bool
    Revcomp        bool
    MaxAmbiguous []int
    Alphabet       string
    Kmers        []string
  }{}
  config.M, config.N  = obj.M, obj.N
  config.Binarize     = obj.Binarize
  config.Complement   = obj.Complement
  config.Reverse      = obj.Reverse
  config.Revcomp      = obj.Revcomp
  config.MaxAmbiguous = obj.MaxAmbiguous
  config.Alphabet     = obj.Alphabet.String()
  config.Kmers        = make([]string, len(obj.Kmers))
  for i, kmer := range obj.Kmers {
    config.Kmers[i] = kmer.String()
  }
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
