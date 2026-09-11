// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package centrality

import (
	"context"
	"log"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/constraints"
)

// Betweenness computes Brandes' betweenness centrality for every vertex of
// the directed graph a: how many shortest paths between other vertex pairs
// pass through it, with pairs that have k shortest paths contributing 1/k
// per path. The graph is treated as unweighted, in the fewest-hops sense:
// edge weights are ignored and only the pattern of a is used, with zero
// marking an absent edge. Scores are raw (unnormalized) and endpoints are
// excluded. Prefer float64: path counts are exact integers only up to 2^53
// (2^24 for float32) and real graphs exceed the float32 range quickly.
func Betweenness[T constraints.Float](ctx context.Context, a graphblas.Matrix[T]) graphblas.Vector[T] {
	sources := make([]int, a.Rows())
	for i := range sources {
		sources[i] = i
	}
	return BetweennessFromSources[T](ctx, a, sources)
}

// BetweennessFromSources is Betweenness restricted to shortest paths that
// start at the given source vertices. Passing a sample of sources gives
// the standard approximation for graphs too large to sweep completely;
// passing every vertex gives the exact scores.
func BetweennessFromSources[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], sources []int) graphblas.Vector[T] {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("Betweenness requires a square matrix, found %+v x %+v", n, a.Columns())
	}

	for _, s := range sources {
		if s < 0 || s >= n {
			log.Panicf("Source '%+v' is invalid", s)
		}
	}

	bcData := make([]T, n)

	// Both propagation directions as CSR patterns with unit weights:
	// forward (the transpose) advances frontiers along edge direction,
	// backward gathers each vertex's out-neighbors.
	var forward, backward []graphblas.Edge[T]
	for iterator := a.Enumerate(); iterator.HasNext(); {
		u, v, w := iterator.Next()
		if w == 0 {
			continue
		}
		forward = append(forward, graphblas.Edge[T]{From: v, To: u, Weight: 1})
		backward = append(backward, graphblas.Edge[T]{From: u, To: v, Weight: 1})
	}
	at := graphblas.NewCSRMatrixFromEdges(n, n, forward)
	ap := graphblas.NewCSRMatrixFromEdges(n, n, backward)

	sigmaData := make([]T, n)
	sigma := graphblas.NewDenseVectorFromArrayN(sigmaData)
	qData := make([]T, n)
	q := graphblas.NewDenseVectorFromArrayN(qData)
	nextData := make([]T, n)
	next := graphblas.NewDenseVectorFromArrayN(nextData)
	t1Data := make([]T, n)
	t1 := graphblas.NewDenseVectorFromArrayN(t1Data)
	t2Data := make([]T, n)
	t2 := graphblas.NewDenseVectorFromArrayN(t2Data)
	deltaData := make([]T, n)

	for _, s := range sources {
		select {
		case <-ctx.Done():
			return graphblas.NewDenseVectorFromArrayN(bcData)
		default:
		}

		// Forward: breadth-first sweeps propagate shortest-path counts.
		// sigma accumulates the count per vertex and doubles as the
		// visited mask, so each vertex is counted at its first depth only.
		clear(sigmaData)
		clear(qData)
		sigmaData[s] = 1
		qData[s] = 1

		levels := [][]T{snapshot(qData)}
		for {
			clear(nextData)
			graphblas.MatrixVectorMultiply[T](ctx, at, q, sigma, next)

			any := false
			for i, v := range nextData {
				if v != 0 {
					sigmaData[i] += v
					any = true
				}
			}
			if !any {
				break
			}

			levels = append(levels, snapshot(nextData))
			copy(qData, nextData)
		}

		// Backward: dependencies flow from the deepest level toward the
		// source. Each level-k vertex offers (1 + delta) / sigma, its
		// level k-1 predecessors gather those offers along their
		// out-edges, and scale by their own path counts.
		clear(deltaData)
		for k := len(levels) - 1; k >= 1; k-- {
			clear(t1Data)
			for i, lv := range levels[k] {
				if lv != 0 {
					t1Data[i] = (1 + deltaData[i]) / sigmaData[i]
				}
			}

			clear(t2Data)
			graphblas.MatrixVectorMultiply[T](ctx, ap, t1, nil, t2)

			for i, lv := range levels[k-1] {
				if lv != 0 {
					deltaData[i] += sigmaData[i] * t2Data[i]
				}
			}
		}

		for i, d := range deltaData {
			if i != s {
				bcData[i] += d
			}
		}
	}

	return graphblas.NewDenseVectorFromArrayN(bcData)
}

func snapshot[T constraints.Float](data []T) []T {
	out := make([]T, len(data))
	copy(out, data)
	return out
}
