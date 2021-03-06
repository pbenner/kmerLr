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
import   "bytes"
import   "sort"
import   "strings"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

type FeatureIndices [][2]int

func newFeatureIndices(n int, cooccurrence bool) FeatureIndices {
  if cooccurrence {
    m := (n+1)*n/2
    features := make(FeatureIndices, m)
    for i1 := 0; i1 < n; i1++ {
      for i2 := i1; i2 < n; i2++ {
        j := CoeffIndex(n).Ind2Sub(i1, i2) - 1
        features[j] = [2]int{i1, i2}
      }
    }
    return features
  } else {
    features := make(FeatureIndices, n)
    for i := 0; i < n; i++ {
      features[i] = [2]int{i, i}
    }
    return features
  }
}

func (obj FeatureIndices) Clone() FeatureIndices {
  r := make(FeatureIndices, len(obj))
  for i, feature := range obj {
    r[i] = [2]int{feature[0], feature[1]}
  }
  return r
}

func (obj FeatureIndices) Less(i, j int) bool {
  if obj[i][0] == obj[i][1] {
    if obj[j][0] == obj[j][1] {
      return obj[i][0] < obj[j][0]
    } else {
      return true
    }
  } else {
    if obj[j][0] == obj[j][1] {
      return false
    } else {
      if obj[i][0] < obj[j][0] {
        return true
      } else {
        return obj[i][1] < obj[j][1]
      }
    }
  }
}

func (obj FeatureIndices) Swap(i, j int) {
  obj[i], obj[j] = obj[j], obj[i]
}

func (obj FeatureIndices) Len() int {
  return len(obj)
}

func (obj FeatureIndices) Sort() {
  sort.Sort(obj)
}

func (obj FeatureIndices) Equals(b FeatureIndices) bool {
  if len(obj) != len(b) {
    return false
  }
  for i := 0; i < len(b); i++ {
    if obj[i][0] != b[i][0] || obj[i][1] != b[i][1] {
      return false
    }
  }
  return true
}

func (obj FeatureIndices) String(kmers KmerClassList) string {
  var buffer bytes.Buffer

  w := bufio.NewWriter(&buffer)
  for _, feature := range obj {
    if feature[0] == feature[1] {
      fmt.Fprintf(w, " %v", kmers[feature[0]])
    } else {
      fmt.Fprintf(w, " %v&%v", kmers[feature[0]], kmers[feature[1]])
    }
  }
  w.Flush()

  return buffer.String()
}

/* -------------------------------------------------------------------------- */

type KmerLrEquivalence struct {
  KmerEquivalence
  Cooccurrence bool
  Binarize     bool
}

func (obj *KmerLrEquivalence) Equals(a KmerLrEquivalence) error {
  if !obj.KmerEquivalence.Equals(a.KmerEquivalence) {
    return fmt.Errorf("alphabet not consistent across classifiers")
  }
  if  obj.Binarize != a.Binarize {
    return fmt.Errorf("data binarization is not consistent across classifiers")
  }
  if  obj.Cooccurrence != a.Cooccurrence {
    return fmt.Errorf("co-occurrence is not consistent across classifiers")
  }
  return nil
}

/* -------------------------------------------------------------------------- */

type KmerLrFeatures struct {
  KmerLrEquivalence
  Kmers    KmerClassList
  Features FeatureIndices
}

/* -------------------------------------------------------------------------- */

func (obj KmerLrFeatures) Clone() KmerLrFeatures {
  r := KmerLrFeatures{}
  r.KmerLrEquivalence = obj.KmerLrEquivalence
  r.Kmers             = obj.Kmers   .Clone()
  r.Features          = obj.Features.Clone()
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *KmerLrFeatures) ImportConfig(config ConfigDistribution, t ScalarType) error {
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
  cooccurrence, ok := config.GetNamedParameterAsBool("Cooccurrence"); if !ok {
    // backward compatibility
    cooccurrence = false
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
  features, ok := config.GetNamedParametersAsIntPairs("Features"); if !ok {
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
  if rel, err := NewKmerEquivalenceRelation(m, n, complement, reverse, revcomp, maxAmbiguous, obj.Alphabet); err != nil {
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
  obj.Cooccurrence = cooccurrence
  obj.Reverse      = reverse
  obj.Revcomp      = revcomp
  obj.MaxAmbiguous = maxAmbiguous
  obj.Features     = features
  return nil
}

func (obj *KmerLrFeatures) ExportConfig() ConfigDistribution {
  config := struct{
    M, N           int
    Binarize       bool
    Complement     bool
    Cooccurrence   bool
    Reverse        bool
    Revcomp        bool
    MaxAmbiguous []int
    Alphabet       string
    Kmers        []string
    Features       FeatureIndices
  }{}
  config.M, config.N  = obj.M, obj.N
  config.Binarize     = obj.Binarize
  config.Complement   = obj.Complement
  config.Cooccurrence = obj.Cooccurrence
  config.Reverse      = obj.Reverse
  config.Revcomp      = obj.Revcomp
  config.MaxAmbiguous = obj.MaxAmbiguous
  config.Features     = obj.Features
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
