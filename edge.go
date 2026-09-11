// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas

import (
	"log"
	"sort"

	"github.com/rossmerr/graphblas/constraints"
)

// Edge is a weighted directed edge for building sparse matrices: Weight is
// stored at row From, column To.
type Edge[T constraints.Number] struct {
	From   int
	To     int
	Weight T
}

// NewCSRMatrixFromEdges builds an r x c CSRMatrix from an edge list in one
// pass over the edges plus a sort within each row, instead of the
// per-element inserts Set performs. Edges may arrive in any order.
// Duplicate edges sum their weights; zero-weight edges, and duplicates
// whose weights sum to zero, are not stored.
func NewCSRMatrixFromEdges[T constraints.Number](r, c int, edges []Edge[T]) *CSRMatrix[T] {
	for _, e := range edges {
		if e.From < 0 || e.From >= r {
			log.Panicf("Row '%+v' is invalid", e.From)
		}

		if e.To < 0 || e.To >= c {
			log.Panicf("Column '%+v' is invalid", e.To)
		}
	}

	s := newCSRMatrix[T](r, c, 0)

	counts := make([]int, r+1)
	nnz := 0
	for _, e := range edges {
		if IsZero(e.Weight) {
			continue
		}
		counts[e.From+1]++
		nnz++
	}

	rowStart := counts
	for i := 0; i < r; i++ {
		rowStart[i+1] += rowStart[i]
	}

	cols := make([]int, nnz)
	values := make([]T, nnz)
	next := make([]int, r)
	copy(next, rowStart[:r])
	for _, e := range edges {
		if IsZero(e.Weight) {
			continue
		}
		cols[next[e.From]] = e.To
		values[next[e.From]] = e.Weight
		next[e.From]++
	}

	// Sort each row by column and merge duplicates by summing, compacting
	// the arrays in place. out never overtakes j, so the compaction is
	// safe.
	out := 0
	newStart := make([]int, r+1)
	for i := 0; i < r; i++ {
		newStart[i] = out
		start, end := rowStart[i], rowStart[i+1]
		sort.Sort(edgeRow[T]{cols: cols[start:end], values: values[start:end]})

		for j := start; j < end; {
			col := cols[j]
			sum := values[j]
			j++
			for j < end && cols[j] == col {
				sum += values[j]
				j++
			}

			if !IsZero(sum) {
				cols[out] = col
				values[out] = sum
				out++
			}
		}
	}
	newStart[r] = out

	s.cols = cols[:out]
	s.values = values[:out]
	s.rowStart = newStart
	return s
}

type edgeRow[T constraints.Number] struct {
	cols   []int
	values []T
}

func (e edgeRow[T]) Len() int           { return len(e.cols) }
func (e edgeRow[T]) Less(i, j int) bool { return e.cols[i] < e.cols[j] }
func (e edgeRow[T]) Swap(i, j int) {
	e.cols[i], e.cols[j] = e.cols[j], e.cols[i]
	e.values[i], e.values[j] = e.values[j], e.values[i]
}
