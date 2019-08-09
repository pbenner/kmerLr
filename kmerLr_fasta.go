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

func convert_counts(config Config, counts KmerCounts, label int) ConstVector {
  n := counts.Len()+1
  if label != -1 {
    n += 1
  }
  i := make([]int      , n)
  v := make([]ConstReal, n)
  i[0] = 0
  v[0] = 1.0
  // copy counts to (i, v)
  for j, it := 1, counts.Iterate(); it.Ok(); it.Next() {
    if c := it.GetCount(); c != 0 {
      i[j] = j
      v[j] = ConstReal(c)
    }
    j++
  }
  if label != -1 {
    // append label to data vector
    i[n-1] = n-1
    v[n-1] = ConstReal(label)
  }
  r := NilSparseConstRealVector(n)
  r.SetSparseIndices(i)
  r.SetSparseValues (v)
  return r
}

func convert_counts_list(config Config, countsList *KmerCountsList, label int) []ConstVector {
  r := make([]ConstVector, countsList.Len())
  PrintStderr(config, 1, "Converting kmer counts... ")
  if err := config.Pool.RangeJob(0, countsList.Len(), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    r[i] = convert_counts(config, countsList.At(i), label)
    // free memory
    countsList.Counts[i] = nil
    return nil
  }); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  return r
}

/* -------------------------------------------------------------------------- */

func scan_sequence(config Config, kmersCounter *KmerCounter, binarize bool, sequence []byte) KmerCounts {
  if binarize {
    return kmersCounter.IdentifyKmers(sequence)
  } else {
    return kmersCounter.CountKmers(sequence)
  }
}

func scan_sequences(config Config, kmersCounter *KmerCounter, binarize bool, sequences []string) []KmerCounts {
  r := make([]KmerCounts, len(sequences))
  // create one counter for each thread
  counters := make([]*KmerCounter, config.Pool.NumberOfThreads())
  for i, _ := range counters {
    counters[i] = kmersCounter.Clone()
  }
  PrintStderr(config, 1, "Counting kmers... ")
  if err := config.Pool.RangeJob(0, len(sequences), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    r[i] = scan_sequence(config, counters[pool.GetThreadId()], binarize, []byte(sequences[i]))
    return nil
  }); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  return r
}

/* -------------------------------------------------------------------------- */

func compile_training_data(config Config, kmersCounter *KmerCounter, binarize bool, filename_fg, filename_bg string) ([]ConstVector, KmerClassList) {
  fg := import_fasta(config, filename_fg)
  bg := import_fasta(config, filename_bg)
  counts_fg := scan_sequences(config, kmersCounter, binarize, fg)
  counts_bg := scan_sequences(config, kmersCounter, binarize, bg)
  counts_list    := NewKmerCountsList(append(counts_fg, counts_bg...)...)
  counts_list_fg := counts_list.Slice(      0, len(fg))
  counts_list_bg := counts_list.Slice(len(fg), len(fg)+len(bg))
  r_fg := convert_counts_list(config, &counts_list_fg, 1)
  r_bg := convert_counts_list(config, &counts_list_bg, 0)
  return append(r_fg, r_bg...), counts_list.Kmers
}

func compile_test_data(config Config, kmersCounter *KmerCounter, kmers KmerClassList, binarize bool, filename string) ([]ConstVector, KmerClassList) {
  sequences   := import_fasta(config, filename)
  counts      := scan_sequences(config, kmersCounter, binarize, sequences)
  counts_list := NewKmerCountsList(counts...)
  // set counts_list.Kmers to the set of kmers on which the
  // classifier was trained on
  counts_list.Kmers = kmers
  return convert_counts_list(config, &counts_list, -1), counts_list.Kmers
}
