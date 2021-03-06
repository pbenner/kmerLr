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
import   "log"
import   "io"
import   "bufio"
import   "bytes"
import   "os"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

/* -------------------------------------------------------------------------- */

type KmerDataSet struct {
  Data   []ConstVector
  Labels []bool
  Kmers    KmerClassList
}

func (obj KmerDataSet) String() string {
  var buffer bytes.Buffer

  w := bufio.NewWriter(&buffer)
  l := make([]int, len(obj.Kmers))
  m := 12
  if len(l) > 0 {
    // compute kmer string lengths
    for j := 0; j < len(l); j++ {
      l[j] = len(obj.Kmers[j].String())
      if l[j] < m {
        l[j] = m
      }
    }
    // print header
    for j := 0; j < len(l); j++ {
      f := fmt.Sprintf("%%%ds ", l[j])
      fmt.Fprintf(w, f, obj.Kmers[j])
    }
    if len(obj.Labels) != 0 {
      fmt.Fprintf(w, "labels")
    }
    fmt.Fprintf(w, "\n")
    // print data
    for i := 0; i < len(obj.Data); i++ {
      for j := 0; j < len(l); j++ {
        f := fmt.Sprintf("%%%de ", l[j])
        fmt.Fprintf(w, f, obj.Data[i].Float64At(j+1))
      }
      if len(obj.Labels) != 0 {
        if obj.Labels[i] {
          fmt.Fprintf(w, "1")
        } else {
          fmt.Fprintf(w, "0")
        }
      }
      fmt.Fprintf(w, "\n")
    }
  } else {
    // print data
    for i := 0; i < len(obj.Data); i++ {
      for j := 0; j < obj.Data[i].Dim()-1; j++ {
        fmt.Fprintf(w, "%12e ", obj.Data[i].Float64At(j+1))
      }
      if len(obj.Labels) != 0 {
        if obj.Labels[i] {
          fmt.Fprintf(w, "1")
        } else {
          fmt.Fprintf(w, "0")
        }
      }
      fmt.Fprintf(w, "\n")
    }
  }
  w.Flush()

  return buffer.String()
}

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

func export_kmers(config Config, filename string, data KmerDataSet) {
  var writer io.Writer
  if filename == "" {
    writer = os.Stdout
  } else {
    f, err := os.Create(filename)
    if err != nil {
      panic(err)
    }
    defer f.Close()

    w := bufio.NewWriter(f)
    defer w.Flush()

    writer = w
  }
  for j := 0; j < len(data.Kmers); j++ {
    if j == 0 {
      fmt.Fprintf(writer,  "%s", data.Kmers[j])
    } else {
      fmt.Fprintf(writer, ",%s", data.Kmers[j])
    }
  }
  fmt.Fprintf(writer, "\n")
  if config.DataTransform != "" {
    for i := 0; i < len(data.Data); i++ {
      for j := 1; j < data.Data[i].Dim(); j++ {
        if j == 1 {
          fmt.Fprintf(writer,  "%e", data.Data[i].Float64At(j))
        } else {
          fmt.Fprintf(writer, ",%e", data.Data[i].Float64At(j))
        }
      }
      fmt.Fprintf(writer, "\n")
    }
  } else {
    for i := 0; i < len(data.Data); i++ {
      for j := 1; j < data.Data[i].Dim(); j++ {
        if j == 1 {
          fmt.Fprintf(writer,  "%d", data.Data[i].IntAt(j))
        } else {
          fmt.Fprintf(writer, ",%d", data.Data[i].IntAt(j))
        }
      }
      fmt.Fprintf(writer, "\n")
    }
  }
}

/* -------------------------------------------------------------------------- */

func compute_class_weights(c []bool) [2]float64 {
  n  := len(c)
  n1 := 0
  n0 := 0
  for i := 0; i < n; i++ {
    if c[i] {
      n1++
    } else {
      n0++
    }
  }
  r := [2]float64{}
  r[0] = float64(n0+n1)/float64(2*n0)
  r[1] = float64(n0+n1)/float64(2*n1)
  return r
}

/* -------------------------------------------------------------------------- */

func convert_counts(config Config, counts KmerCounts, features FeatureIndices, generate_features bool) ConstVector {
  n := 0
  i := []int    {0  }
  v := []float64{1.0}
  if len(features) == 0 && generate_features {
    n = counts.Len()+1
    // copy counts to (i, v)
    for it := counts.Iterate(); it.Ok(); it.Next() {
      if c := it.GetCount(); c != 0 {
        i = append(i, it.GetIndex()+1)
        v = append(v, float64(c))
      }
    }
  } else {
    n = len(features)+1
    for j, feature := range features {
      i1 := feature[0]
      i2 := feature[1]
      if i1 == i2 {
        c := counts.Counts[counts.Kmers[i1].KmerClassId]
        if c != 0.0 {
          i = append(i, j+1)
          v = append(v, float64(c))
        }
      } else {
        c1 := counts.Counts[counts.Kmers[i1].KmerClassId]
        c2 := counts.Counts[counts.Kmers[i2].KmerClassId]
        if c1 != 0.0 && c2 != 0.0 {
          i = append(i, j+1)
          v = append(v, float64(c1*c2))
        }
      }
    }
  }
  // resize slice and restrict capacity
  i = append([]int    {}, i[0:len(i)]...)
  v = append([]float64{}, v[0:len(v)]...)
  return UnsafeSparseConstFloat64Vector(i, v, n)
}

func convert_counts_list(config Config, countsList *KmerCountsList, features FeatureIndices, generate_features bool) []ConstVector {
  r := make([]ConstVector, countsList.Len())
  PrintStderr(config, 1, "Converting kmer counts... ")
  if err := config.Pool.RangeJob(0, countsList.Len(), func(i int, pool threadpool.ThreadPool, erf func() error) error {
    config := config; config.Pool = pool

    r[i] = convert_counts(config, countsList.At(i), features, generate_features)
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
    config := config; config.Pool = pool

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

func reduce_samples(config Config, fg, bg []string) ([]string, []string) {
  if config.MaxSamples == 0 || len(fg) + len(bg) <= config.MaxSamples {
    return fg, bg
  }
  n1 := len(fg)
  n2 := len(bg)
  m1 := int(float64(n1)/float64(n1+n2)*float64(config.MaxSamples))
  m2 := config.MaxSamples - m1
  if m1 <= 0 || m2 <= 0 {
    log.Fatal("cannot reduce samples")
  } else {
    PrintStderr(config, 1, "Reduced training data from (%d,%d) to (%d,%d) samples\n", n1, n2, m1, m2)
  }
  return fg[0:m1], bg[0:m2]
}

/* -------------------------------------------------------------------------- */

func compile_training_data(config Config, kmersCounter *KmerCounter, kmers KmerClassList, features FeatureIndices, generate_features bool, binarize bool, filename_fg, filename_bg string) KmerDataSet {
  fg := import_fasta(config, filename_fg)
  bg := import_fasta(config, filename_bg)
  fg, bg  = reduce_samples(config, fg, bg)
  labels := make([]bool, len(fg)+len(bg))
  for i := 0; i < len(fg); i++ {
    labels[i] = true
  }
  counts_fg   := scan_sequences(config, kmersCounter, binarize, fg)
  counts_bg   := scan_sequences(config, kmersCounter, binarize, bg)
  counts_list := NewKmerCountsList(append(counts_fg, counts_bg...)...)
  if len(kmers) != 0 {
    counts_list.SetKmers(kmers)
  }
  counts_list_fg := counts_list.Slice(      0, len(fg))
  counts_list_bg := counts_list.Slice(len(fg), len(fg)+len(bg))
  r_fg := convert_counts_list(config, &counts_list_fg, features, generate_features)
  r_bg := convert_counts_list(config, &counts_list_bg, features, generate_features)
  return KmerDataSet{append(r_fg, r_bg...), labels, counts_list.Kmers}
}

func compile_test_data(config Config, kmersCounter *KmerCounter, kmers KmerClassList, features FeatureIndices, generate_features bool, binarize bool, filename string) KmerDataSet {
  sequences   := import_fasta(config, filename)
  counts      := scan_sequences(config, kmersCounter, binarize, sequences)
  counts_list := NewKmerCountsList(counts...)
  // set counts_list.Kmers to the set of kmers on which the
  // classifier was trained on
  counts_list.SetKmers(kmers)
  return KmerDataSet{convert_counts_list(config, &counts_list, features, generate_features), nil, counts_list.Kmers}
}

/* -------------------------------------------------------------------------- */

func compile_data(config Config, kmersCounter *KmerCounter, kmers KmerClassList, features FeatureIndices, generate_features bool, binarize bool, filenames []string) []KmerDataSet {
  r := make([]KmerDataSet, len(filenames))
  c := make([]KmerCounts , 0)
  k := make([]int        , len(filenames)+1)
  for i, filename := range filenames {
    seq   := import_fasta(config, filename)
    c      = append(c, scan_sequences(config, kmersCounter, binarize, seq)...)
    k[i+1] = len(c)
  }
  counts_list := NewKmerCountsList(c...)
  if len(kmers) != 0 {
    counts_list.SetKmers(kmers)
  }
  for i, _ := range filenames {
    tmp := counts_list.Slice(k[i], k[i+1])
    r[i] = KmerDataSet{convert_counts_list(config, &tmp, features, generate_features), nil, counts_list.Kmers}
  }
  return r
}
