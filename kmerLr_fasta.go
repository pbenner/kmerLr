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
import   "log"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func import_fasta(config Config, filename string) []string {
  s := OrderedStringSet{}
  PrintStderr(config, 1, "Reading fasta file `%s'... ", filename)
  if err := s.ImportFasta(filename); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  r := make([]string, len(s.Seqnames))
  for i, name := range s.Seqnames {
    r[i] = string(s.Sequences[name])
  }
  return r
}

/* -------------------------------------------------------------------------- */

func scan_sequence(config Config, kmersCounter KmersCounter, binarize bool, label int, sequence []byte) (ConstVector, error) {
  i := []int(nil)
  t := []int(nil)
  v := []float64(nil)
  n := kmersCounter.Length()
  if binarize {
    i = kmersCounter.IdentifyKmersSparse(sequence)
    v = make([]float64, len(i))
    for k := 0; k < len(v); k++ {
      v[k] = 1.0
    }
  } else {
    i, t = kmersCounter.CountKmersSparse(sequence)
    v = make([]float64, len(i))
    for k := 0; k < len(v); k++ {
      v[k] = float64(t[k])
    }
  }
  { // append 1 to data vector
    n += 1
    for k := 0; k < len(i); k++ {
      i[k] += 1
    }
    i = append([]    int{0  }, i...)
    v = append([]float64{1.0}, v...)
  }
  if label != -1 {
    // append label to data vector
    n += 1
    i = append(i, n-1)
    v = append(v, float64(label))
  }
  return NewSparseConstRealVector(i, v, n), nil
}

func scan_sequences(config Config, kmersCounter KmersCounter, binarize bool, label int, sequences []string) []ConstVector {
  result := make([]ConstVector, len(sequences))

  PrintStderr(config, 1, "Counting kmers... ")
  for i, s := range sequences {
    if r, err := scan_sequence(config, kmersCounter, binarize, label, []byte(s)); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    } else {
      result[i] = r
    }
  }
  PrintStderr(config, 1, "done\n")
  return result
}

/* -------------------------------------------------------------------------- */

func compile_training_data(config Config, kmersCounter KmersCounter, binarize bool, filename_fg, filename_bg string) []ConstVector {
  fg := import_fasta(config, filename_fg)
  bg := import_fasta(config, filename_bg)
  fg_counts := scan_sequences(config, kmersCounter, binarize, 1, fg)
  bg_counts := scan_sequences(config, kmersCounter, binarize, 0, bg)
  return append(fg_counts, bg_counts...)
}

func compile_test_data(config Config, kmersCounter KmersCounter, binarize bool, filename string) []ConstVector {
  sequences := import_fasta(config, filename)
  return scan_sequences(config, kmersCounter, binarize, -1, sequences)
}
