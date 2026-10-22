// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering

import (
	"context"
	"log"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/constraints"
)

// DefaultPeerPressureIterations caps the voting rounds when the labels
// have not settled.
const DefaultPeerPressureIterations = 50

// PeerPressure clusters the vertices of the directed graph a by peer
// pressure: every vertex starts in its own cluster, then each round every
// vertex votes for its out-neighbors to join its cluster and each vertex
// moves to the cluster with the most votes, until no vertex moves. Each
// vertex casts one vote in total, split across its out-edges in
// proportion to weight plus a self-loop weighted like its heaviest
// out-edge, so high-degree vertices do not dominate and a vertex leans
// toward staying put. The tally is one sparse mxm per round, votes times
// the vertex-to-cluster assignment matrix, and ties go to the cluster
// with the lowest label. Voting starts from singletons and is local, so a
// sparse community can settle split across several clusters; peer
// pressure suits graphs whose communities are dense. a.At(u, v) holds the
// weight of edge u -> v, zero marks an absent edge, and weights must be
// non-negative. Undirected graphs list each edge in both directions.
// Labels are numbered as the package documentation describes.
func PeerPressure[T constraints.Float](ctx context.Context, a graphblas.Matrix[T]) []int {
	return PeerPressureWithOptions[T](ctx, a, DefaultPeerPressureIterations)
}

// PeerPressureWithOptions is PeerPressure with an explicit cap on voting
// rounds.
func PeerPressureWithOptions[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], maxIterations int) []int {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("PeerPressure requires a square matrix, found %+v x %+v", n, a.Columns())
	}

	total := make([]T, n)
	heaviest := make([]T, n)
	var edges []graphblas.Edge[T]
	for iterator := a.Enumerate(); iterator.HasNext(); {
		u, v, w := iterator.Next()
		if w == 0 {
			continue
		}
		total[u] += w
		heaviest[u] = max(heaviest[u], w)
		edges = append(edges, graphblas.Edge[T]{From: u, To: v, Weight: w})
	}
	for u := range n {
		if heaviest[u] == 0 {
			heaviest[u] = 1
		}
		total[u] += heaviest[u]
		edges = append(edges, graphblas.Edge[T]{From: u, To: u, Weight: heaviest[u]})
	}

	// votes[v][u] is the share of u's vote that goes to v, so row v of
	// votes times the assignment matrix tallies v's votes per cluster.
	for i, e := range edges {
		edges[i] = graphblas.Edge[T]{From: e.To, To: e.From, Weight: e.Weight / total[e.From]}
	}
	votes := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	labels := make([]int, n)
	for v := range labels {
		labels[v] = v
	}

	assignment := make([]graphblas.Edge[T], n)
	for iteration := 0; iteration < maxIterations; iteration++ {
		select {
		case <-ctx.Done():
			return canonical(labels)
		default:
		}

		for v, l := range labels {
			assignment[v] = graphblas.Edge[T]{From: v, To: l, Weight: 1}
		}
		tally := graphblas.NewCSRMatrix[T](n, n)
		graphblas.MatrixMatrixMultiply[T](ctx, votes, graphblas.NewCSRMatrixFromEdges(n, n, assignment), nil, tally)

		// Rows enumerate in column order, so the first maximum seen is
		// the lowest-labeled one. Every vertex votes for itself, so every
		// row has an entry.
		next := make([]int, n)
		best := make([]T, n)
		for iterator := tally.Enumerate(); iterator.HasNext(); {
			v, cluster, count := iterator.Next()
			if count > best[v] {
				best[v] = count
				next[v] = cluster
			}
		}

		moved := false
		for v := range labels {
			if next[v] != labels[v] {
				moved = true
				break
			}
		}
		labels = next
		if !moved {
			break
		}
	}

	return canonical(labels)
}
