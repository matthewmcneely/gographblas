// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package shortestpath

import (
	"context"
	"log"
	"math"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/constraints"
)

// SingleSource computes the shortest-path distance from source vertex s to
// every vertex of the weighted directed graph a using Bellman-Ford edge
// relaxation. a.At(u, v) holds the weight of edge u->v and zero marks an
// absent edge, so explicit zero-weight edges are not representable.
// Unreachable vertices report +Inf. Negative edge weights are supported on
// graphs without negative cycles; when a negative cycle is reachable from
// s, the distances of vertices on or downstream of the cycle are not
// meaningful.
func SingleSource[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], s int) graphblas.Vector[T] {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("SingleSource requires a square matrix, found %+v x %+v", n, a.Columns())
	}

	if s < 0 || s >= n {
		log.Panicf("Source '%+v' is invalid", s)
	}

	inf := T(math.Inf(1))
	dist := graphblas.NewDenseVectorN[T](n)
	for i := 0; i < n; i++ {
		dist.SetVec(i, inf)
	}
	dist.SetVec(s, 0)

	// A shortest path uses at most n-1 edges; stop earlier once a full
	// relaxation round changes nothing.
	for round := 1; round < n; round++ {
		select {
		case <-ctx.Done():
			return dist
		default:
		}

		changed := false
		for iterator := a.Enumerate(); iterator.HasNext(); {
			u, v, w := iterator.Next()
			if w == 0 {
				continue
			}

			du := dist.AtVec(u)
			if du == inf {
				continue
			}

			if d := du + w; d < dist.AtVec(v) {
				dist.SetVec(v, d)
				changed = true
			}
		}

		if !changed {
			break
		}
	}

	return dist
}
