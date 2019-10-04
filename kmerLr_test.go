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
import   "testing"

import . "github.com/pbenner/gonetics"

/* -------------------------------------------------------------------------- */

func Test1(test *testing.T) {
  config := Config{}
  //config.Binarize = true
  config.Verbose  = 2 

  kmersCounter, err := NewKmerCounter(4, 8, false, false, true, nil, GappedNucleotideAlphabet{}); if err != nil {
    test.Error(err)
  } else {
    data1, _, kmers := compile_training_data(config, kmersCounter, nil, nil, "kmerLr_test.fa", "kmerLr_test.fa")

    features := FeatureIndices{}
    for i, _ := range kmers {
      features = append(features, [2]int{i, i})
    }
    features = append(features, [2]int{4671 , 4672 }) // gntanc|gntanc     = 3, gntcaa|ttganc     = 0
    features = append(features, [2]int{5068 , 5486 }) // aaagaaa|tttcttt   = 1, aagannt|anntctt   = 7
    features = append(features, [2]int{19270, 57071}) // aacgcgna|tncgcgtt = 1, tgaatgca|tgcattca = 1
    features = append(features, [2]int{4671 , 5486 }) // gntanc|gntanc     = 3, aagannt|anntctt   = 7
    
    data2, _, kmers := compile_training_data(config, kmersCounter, kmers, features, "kmerLr_test.fa", "kmerLr_test.fa")

    for i, _ := range features[0:len(features)-4] {
      if data1[0].ValueAt(i+1) != data2[0].ValueAt(i+1) {
        test.Error("test failed")
      }
    }
    if data2[0].ValueAt(len(features)-3) != 0 {
      test.Error("test failed")
    }
    if data2[0].ValueAt(len(features)-2) != 7 {
      test.Error("test failed")
    }
    if data2[0].ValueAt(len(features)-1) != 1 {
      test.Error("test failed")
    }
    if data2[0].ValueAt(len(features)-0) != 21 {
      test.Error("test failed")
    }
  }
}
