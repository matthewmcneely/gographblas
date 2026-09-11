// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package centrality

import (
	"context"
	"log"
	"math"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/binaryop"
	"github.com/rossmerr/graphblas/constraints"
)

// Closeness computes closeness centrality for every vertex of the weighted
// directed graph a: the reciprocal of the average shortest-path distance
// from the vertex to the vertices it reaches, scaled by the fraction of
// the graph reached (the Wasserman-Faust adjustment, which keeps scores
// comparable when the graph is not strongly connected). Distances follow
// edge direction and come from min-plus semiring rounds against the
// transposed graph, one Bellman-Ford relaxation per round, so weights
// should be non-negative for the scores to be meaningful. a.At(u, v)
// holds the weight of edge u -> v and zero marks an absent edge. A vertex
// that reaches nothing scores 0.
func Closeness[T constraints.Float](ctx context.Context, a graphblas.Matrix[T]) graphblas.Vector[T] {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("Closeness requires a square matrix, found %+v x %+v", n, a.Columns())
	}

	scores := graphblas.NewDenseVectorN[T](n)
	if n <= 1 {
		return scores
	}

	at := graphblas.TransposeToCSR(ctx, a)
	minPlus := binaryop.MinPlus[T]()
	inf := T(math.Inf(1))

	distData := make([]T, n)
	dist := graphblas.NewDenseVectorFromArrayN(distData)
	nextData := make([]T, n)
	next := graphblas.NewDenseVectorFromArrayN(nextData)

	for s := 0; s < n; s++ {
		select {
		case <-ctx.Done():
			return scores
		default:
		}

		for i := range distData {
			distData[i] = inf
		}
		distData[s] = 0

		for round := 1; round < n; round++ {
			graphblas.MatrixVectorMultiplyWithSemiring[T](ctx, at, dist, nil, next, minPlus)

			changed := false
			for i, v := range nextData {
				if v < distData[i] {
					distData[i] = v
					changed = true
				}
			}

			if !changed {
				break
			}
		}

		sum := T(0)
		reach := 0
		for i, d := range distData {
			if i != s && d < inf {
				sum += d
				reach++
			}
		}

		if reach > 0 && sum > 0 {
			r := T(reach)
			scores.SetVec(s, (r/sum)*(r/T(n-1)))
		}
	}

	return scores
}
