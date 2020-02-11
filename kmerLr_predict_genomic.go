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
import   "io"
import   "log"
import   "math"
import   "os"
import   "strings"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/gonetics"
import   "github.com/pbenner/threadpool"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

func saveWindowPredictionsWiggle(filename string, regions GRanges, predictions [][]float64, track_name string, window_size, window_step int) {
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
  fmt.Fprintf(writer, "track type=wiggle_0 name=%s\n", track_name)
  for i := 0; i < len(predictions); i++ {
    fmt.Fprintf(writer, "fixedStep chrom=%s start=%d step=%d span=%d\n", regions.Seqnames[i], regions.Ranges[i].From+window_size/2, window_step, window_step)
    for j := 0; j < len(predictions[i]); j++ {
      fmt.Fprintf(writer, "%0.15f\n", math.Exp(predictions[i][j]))
    }
  }
}

/* -------------------------------------------------------------------------- */

func importBed3(config Config, filename string) GRanges {
  granges := GRanges{}
  if filename == "" {
    if err := granges.ReadBed3(os.Stdin); err != nil {
      log.Fatal(err)
    }
  } else {
    PrintStderr(config, 1, "Reading bed file `%s'... ", filename)
    if err := granges.ImportBed3(filename); err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    } else {
      PrintStderr(config, 1, "done\n")
    }
  }
  return granges
}

func importFasta(config Config, filename string) StringSet {
  s := StringSet{}
  PrintStderr(config, 1, "Reading fasta file `%s'... ", filename)
  if err := s.ImportFasta(filename); err != nil {
    PrintStderr(config, 1, "failed\n")
    log.Fatal(err)
  }
  PrintStderr(config, 1, "done\n")
  return s
}

func extractFasta(config Config, filenameFasta string, regions GRanges) []string {
  genomicSequence := importFasta(config, filenameFasta)
  // import bed file first to check if it exists
  sequences := make([]string, regions.Length())

  for i := 0; i < regions.Length(); i++ {
    sequence, err := genomicSequence.GetSlice(regions.Seqnames[i], regions.Ranges[i])
    // if sequence is nil, it means the fasta file is missing a chromosome
    if sequence == nil {
      log.Fatalf("sequence `%s' not found in fasta file", regions.Seqnames[i])
    }
    // if squence is not nil but there is an error then the region is out of bounds,
    // GetSlice() then returns only that part which actually exists
    if err != nil {
      log.Fatal(err.Error())
    }
    sequences[i] = string(sequence)
  }
  return sequences
}

/* -------------------------------------------------------------------------- */

type jointKmerLr struct {
  classifiers []*KmerLr
  counters    []*KmerCounter
  configs     []Config
}

func importJointKmerLr(config Config, filename_json string) jointKmerLr {
  filenames   := strings.Split(filename_json, ",")
  configs     := make([]Config      , len(filenames))
  counters    := make([]*KmerCounter, len(filenames))
  classifiers := make([]*KmerLr     , len(filenames))
  for i, filename := range filenames {
    configs    [i] = config
    classifiers[i] = ImportKmerLr(&configs[i], filename)
    if counter, err := NewKmerCounter(configs[i].M, configs[i].N, configs[i].Complement, configs[i].Reverse, configs[i].Revcomp, configs[i].MaxAmbiguous, configs[i].Alphabet, classifiers[i].Kmers...); err != nil {
      log.Fatal(err)
    } else {
      counters[i] = counter
    }
      
  }
  return jointKmerLr{classifiers, counters, configs}
}

func (obj jointKmerLr) Predict(subseq []byte) float64 {
  r := 0.0
  for i, _ := range obj.classifiers {
    counts := scan_sequence(obj.configs[i], obj.counters[i], subseq)
    counts.SetKmers(obj.classifiers[i].Kmers)
    data   := convert_counts(obj.configs[i], counts, obj.classifiers[i].Features)
    r      += obj.classifiers[i].Predict(obj.configs[i], []ConstVector{data})[0]
  }
  return r
}

/* -------------------------------------------------------------------------- */

func predict_window_genomic(config Config, filename_json, filename_fa, filename_bed, filename_out, track_name string, window_size, window_step int) {
  classifier  := importJointKmerLr(config, filename_json)
  regions     := importBed3       (config, filename_bed )
  sequences   := extractFasta     (config, filename_fa, regions)
  predictions := make([][]float64, len(sequences))
  for i, sequence := range sequences {
    if n := len(sequence)-window_size; n > 0 {
      predictions[i] = make([]float64, n/window_step+1)
    }
  }
  job_group := config.Pool.NewJobGroup()
  for i, _ := range sequences {
    i        := i
    sequence := sequences[i]
    for j := 0; j < len(sequence)-window_size; j += window_step {
      j := j
      config.Pool.AddJob(job_group, func(pool threadpool.ThreadPool, erf func() error) error {
        predictions[i][j/window_step] = classifier.Predict([]byte(sequence[j:j+window_size]))
        return nil
      })
    }
  }
  config.Pool.Wait(job_group)

  saveWindowPredictionsWiggle(filename_out, regions, predictions, track_name, window_size, window_step)
}

/* -------------------------------------------------------------------------- */

func main_predict_genomic(config Config, args []string) {
  options := getopt.New()

  optTrackName         := options.StringLong("track-name",           0 , "kmerLr", "name of output track")
  optSlidingWindowSize := options.   IntLong("sliding-window-size",  0 ,      100, "size of the sliding window [default: 100]")
  optSlidingWindowStep := options.   IntLong("sliding-window-step",  0 ,        1, "step size for sliding window [default: 1]")
  optHelp              := options.  BoolLong("help",                'h',           "print help")

  options.SetParameters("<MODEL1.json,MODEL2.json,...> <SEQUENCES.fa> <REGIONS.bed> [RESULT.wig]")
  options.Parse(args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  // parse arguments
  //////////////////////////////////////////////////////////////////////////////
  if len(options.Args()) != 3 && len(options.Args()) != 4 {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  filename_json := options.Args()[0]
  filename_fa   := options.Args()[1]
  filename_bed  := options.Args()[2]
  filename_out  := ""
  if len(options.Args()) == 4 {
    filename_out = options.Args()[3]
  }
  predict_window_genomic(config, filename_json, filename_fa, filename_bed, filename_out, *optTrackName, *optSlidingWindowSize, *optSlidingWindowStep)
}
