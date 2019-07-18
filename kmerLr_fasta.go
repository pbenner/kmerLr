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
import   "github.com/pbenner/threadpool"

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

func convert_counts(config Config, counts KmerCounts, binarize bool, label int) ConstVector {
  n := counts.N()+1
  if label != -1 {
    n += 1
  }
  i := make([]int    , n)
  v := make([]float64, n)
  i[0] = 0
  v[0] = 1.0
  // copy counts to (i, v)
  for j, it := 1, counts.Iterate(); it.Ok(); it.Next() {
    if c := it.GetCount(); c != 0 {
      i[j] = j
      v[j] = float64(c)
    }
    j++
  }
  if label != -1 {
    // append label to data vector
    i[n-1] = n-1
    v[n-1] = float64(label)
  }
  return NewSparseConstRealVector(i, v, n)
}

func convert_counts_list(config Config, countsList KmerCountsList, binarize bool, label int) []ConstVector {
  r := make([]ConstVector, countsList.Len())
  for i := 0; i < countsList.Len(); i++ {
    r[i] = convert_counts(config, countsList.At(i), binarize, label)
    // free memory
    countsList.Counts[i] = nil
  }
  return r
}

/* -------------------------------------------------------------------------- */

func scan_sequence(config Config, kmersCounter *KmersCounter, binarize bool, label int, sequence []byte) KmerCounts {
  if binarize {
    return kmersCounter.IdentifyKmers(sequence)
  } else {
    return kmersCounter.CountKmers(sequence)
  }
}

func scan_sequences(config Config, kmersCounter *KmersCounter, binarize bool, label int, sequences []string) []ConstVector {
  r := make([]KmerCounts, len(sequences))

  PrintStderr(config, 1, "Counting kmers... ")
  if err := config.Pool.RangeJob(0, len(sequences), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    r[i] = scan_sequence(config, kmersCounter, binarize, label, []byte(sequences[i]))
    return nil
  }); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  return convert_counts_list(config, NewKmerCountsList(r...), binarize, label)
}

/* -------------------------------------------------------------------------- */

func compile_training_data(config Config, kmersCounter *KmersCounter, binarize bool, filename_fg, filename_bg string) []ConstVector {
  fg := import_fasta(config, filename_fg)
  bg := import_fasta(config, filename_bg)
  fg_counts := scan_sequences(config, kmersCounter, binarize, 1, fg)
  bg_counts := scan_sequences(config, kmersCounter, binarize, 0, bg)
  return append(fg_counts, bg_counts...)
}

func compile_test_data(config Config, kmersCounter *KmersCounter, binarize bool, filename string) []ConstVector {
  sequences := import_fasta(config, filename)
  return scan_sequences(config, kmersCounter, binarize, -1, sequences)
}
