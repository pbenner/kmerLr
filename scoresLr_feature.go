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

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/statistics"

/* -------------------------------------------------------------------------- */

type ScoresLrFeatures struct {
  Cooccurrence   bool
  Features       FeatureIndices
  Index        []int
  Names        []string
}

/* -------------------------------------------------------------------------- */

func (obj ScoresLrFeatures) Clone() ScoresLrFeatures {
  r := ScoresLrFeatures{}
  r.Cooccurrence    = obj.Cooccurrence
  r.Features        = make(FeatureIndices, len(obj.Features))
  r.Index           = make([]int,    len(obj.Index))
  r.Names           = make([]string, len(obj.Names))
  for i, feature := range obj.Features {
    r.Features[i] = [2]int{feature[0], feature[1]}
  }
  copy(r.Index, obj.Index)
  copy(r.Names, obj.Names)
  return r
}

/* -------------------------------------------------------------------------- */

func (obj *ScoresLrFeatures) ImportConfig(config ConfigDistribution, t ScalarType) error {
  cooccurrence, ok := config.GetNamedParameterAsBool("Cooccurrence"); if !ok {
    // backward compatibility
    cooccurrence = false
  }
  features, ok := config.GetNamedParametersAsIntPairs("Features"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  index, ok := config.GetNamedParametersAsInts("Index"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  names, ok := config.GetNamedParametersAsStrings("Names"); if !ok {
    return fmt.Errorf("invalid config file")
  }
  obj.Cooccurrence = cooccurrence
  obj.Features     = features
  obj.Index        = index
  obj.Names        = names
  return nil
}

func (obj *ScoresLrFeatures) ExportConfig() ConfigDistribution {
  config := struct{
    Cooccurrence   bool
    Features       FeatureIndices
    Index        []int
    Names        []string
  }{}
  config.Cooccurrence = obj.Cooccurrence
  config.Features     = obj.Features
  config.Index        = obj.Index
  config.Names        = obj.Names
  return NewConfigDistribution("features-scores", config)
}
