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
import   "os"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

func import_fasta(config Config, filename string) []string {
  s := OrderedStringSet{}
  if filename == "" {
    PrintStderr(config, 1, "Reading fasta file from stdin... ")
    if err := s.ReadFasta(os.Stdin); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    }
    PrintStderr(config, 1, "done\n")
  } else {
    PrintStderr(config, 1, "Reading fasta file `%s'... ", filename)
    if err := s.ImportFasta(filename); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    }
    PrintStderr(config, 1, "done\n")
  }
  r := make([]string, len(s.Seqnames))
  for i, name := range s.Seqnames {
    r[i] = string(s.Sequences[name])
  }
  return r
}

/* -------------------------------------------------------------------------- */

func compute_class_weights(data []ConstVector) [2]float64 {
  n  := len(data)
  n1 := 0
  n0 := 0
  for i := 0; i < n; i++ {
    _, v := data[i].(SparseConstRealVector).Last()
    switch v {
    case 1.0: n1++
    case 0.0: n0++
    default:
      panic("internal error")
    }
  }
  r := [2]float64{}
  r[0] = float64(n0+n1)/float64(2*n0)
  r[1] = float64(n0+n1)/float64(2*n1)
  return r
}

/* -------------------------------------------------------------------------- */

func convert_counts(config Config, counts KmerCounts, label int) ConstVector {
  n := counts.Len()+1
  m := counts.N  ()+1
  if config.Cooccurrence {
    n = (counts.Len()+1)*counts.Len()/2 + 1
    m = (counts.N  ()+1)*counts.N  ()/2 + 1
  }
  if label != -1 {
    n += 1
    m += 1
  }
  j := 1
  i := make([]int      , m)
  v := make([]ConstReal, m)
  i[0] = 0
  v[0] = 1.0
  // copy counts to (i, v)
  for it := counts.Iterate(); it.Ok(); it.Next() {
    if c := it.GetCount(); c != 0 {
      i[j] = it.GetIndex()+1
      v[j] = ConstReal(c)
      j++
    }
  }
  if config.Cooccurrence {
    p := counts.Len()
    q := j
    for j1 := 1; j1 < q; j1++ {
      for j2 := j1+1; j2 < q; j2++ {
        i1   := i[j1]-1
        i2   := i[j2]-1
        i[j]  = CoeffIndex(p).Ind2Sub(i1, i2)
        v[j]  = v[j1]*v[j2]
        j++
      }
    }
  }
  if label != -1 {
    // append label to data vector
    i[j] = n-1
    v[j] = ConstReal(label)
    j++
  }
  return UnsafeSparseConstRealVector(i[0:j], v[0:j], n)
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

func scan_sequence(config Config, kmersCounter *KmerCounter, sequence []byte) KmerCounts {
  if config.Binarize {
    return kmersCounter.IdentifyKmers(sequence)
  } else {
    return kmersCounter.CountKmers(sequence)
  }
}

func scan_sequences(config Config, kmersCounter *KmerCounter, sequences []string) []KmerCounts {
  r := make([]KmerCounts, len(sequences))
  // create one counter for each thread
  counters := make([]*KmerCounter, config.Pool.NumberOfThreads())
  for i, _ := range counters {
    counters[i] = kmersCounter.Clone()
  }
  PrintStderr(config, 1, "Counting kmers... ")
  if err := config.Pool.RangeJob(0, len(sequences), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    r[i] = scan_sequence(config, counters[pool.GetThreadId()], []byte(sequences[i]))
    return nil
  }); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  return r
}

/* -------------------------------------------------------------------------- */

func compile_training_data(config Config, kmersCounter *KmerCounter, kmers KmerClassList, filename_fg, filename_bg string) ([]ConstVector, KmerClassList) {
  fg := import_fasta(config, filename_fg)
  bg := import_fasta(config, filename_bg)
  counts_fg   := scan_sequences(config, kmersCounter, fg)
  counts_bg   := scan_sequences(config, kmersCounter, bg)
  counts_list := NewKmerCountsList(append(counts_fg, counts_bg...)...)
  if len(kmers) != 0 {
    counts_list.Kmers = kmers
  }
  counts_list_fg := counts_list.Slice(      0, len(fg))
  counts_list_bg := counts_list.Slice(len(fg), len(fg)+len(bg))
  r_fg := convert_counts_list(config, &counts_list_fg, 1)
  r_bg := convert_counts_list(config, &counts_list_bg, 0)
  return append(r_fg, r_bg...), counts_list.Kmers
}

func compile_test_data(config Config, kmersCounter *KmerCounter, kmers KmerClassList, filename string) ([]ConstVector, KmerClassList) {
  sequences   := import_fasta(config, filename)
  counts      := scan_sequences(config, kmersCounter, sequences)
  counts_list := NewKmerCountsList(counts...)
  // set counts_list.Kmers to the set of kmers on which the
  // classifier was trained on
  counts_list.Kmers = kmers
  return convert_counts_list(config, &counts_list, -1), counts_list.Kmers
}
