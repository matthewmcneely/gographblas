// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

// Package clustering partitions the vertices of a graph into clusters.
// Markov, PeerPressure and Spectral label every vertex, returning one
// cluster index per vertex, numbered from 0 in order of each cluster's
// lowest vertex so equal partitions always produce equal labels. Local
// instead grows a single cluster around a seed vertex and returns its
// members. Groups turns labels into member lists.
package clustering

import (
	"log"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/constraints"
)

// Groups lists the members of each cluster, in vertex order, indexed by
// the labels returned from Markov, PeerPressure or Spectral.
func Groups(labels []int) [][]int {
	var groups [][]int
	for v, l := range labels {
		for len(groups) <= l {
			groups = append(groups, nil)
		}
		groups[l] = append(groups[l], v)
	}
	return groups
}

// canonical renumbers labels in place so clusters are numbered from 0 in
// order of their lowest vertex.
func canonical(labels []int) []int {
	renumber := make(map[int]int)
	for v, l := range labels {
		id, ok := renumber[l]
		if !ok {
			id = len(renumber)
			renumber[l] = id
		}
		labels[v] = id
	}
	return labels
}

// symmetrize returns the undirected form of the square matrix a as a CSR
// matrix w = a + a', together with each vertex's weighted degree (the row
// sums of w). Edges given in both directions are counted twice, which
// scales every weight uniformly and leaves degree-normalized quantities
// unchanged.
func symmetrize[T constraints.Float](name string, a graphblas.Matrix[T]) (*graphblas.CSRMatrix[T], []T) {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("%s requires a square matrix, found %+v x %+v", name, n, a.Columns())
	}

	var edges []graphblas.Edge[T]
	for iterator := a.Enumerate(); iterator.HasNext(); {
		u, v, w := iterator.Next()
		if w == 0 {
			continue
		}
		edges = append(edges, graphblas.Edge[T]{From: u, To: v, Weight: w}, graphblas.Edge[T]{From: v, To: u, Weight: w})
	}

	w := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	degree := make([]T, n)
	for iterator := w.Enumerate(); iterator.HasNext(); {
		u, _, weight := iterator.Next()
		degree[u] += weight
	}

	return w, degree
}
