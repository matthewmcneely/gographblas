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

const (
	// DefaultDamping is the probability a random surfer follows an
	// out-link instead of teleporting to a random vertex.
	DefaultDamping = 0.85
	// DefaultTolerance is the L1 threshold under which the rank vector
	// is considered converged.
	DefaultTolerance = 1e-6
	// DefaultIterations caps the power iteration when the tolerance is
	// not reached.
	DefaultIterations = 100
)

// PageRank ranks the vertices of the directed graph a by the stationary
// probability that a random surfer occupies them. a.At(u, v) holds the
// weight of edge u->v and zero marks an absent edge; a vertex spreads its
// rank across its out-links in proportion to edge weight. The rank held by
// dangling vertices (no out-links) is redistributed uniformly, so the
// returned ranks sum to 1. Uses DefaultDamping, DefaultTolerance and
// DefaultIterations.
func PageRank[T constraints.Float](ctx context.Context, a graphblas.Matrix[T]) graphblas.Vector[T] {
	return PageRankWithOptions[T](ctx, a, DefaultDamping, DefaultTolerance, DefaultIterations)
}

// PageRankWithOptions is PageRank with an explicit damping factor, L1
// convergence tolerance and iteration cap.
func PageRankWithOptions[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], damping, tolerance T, maxIterations int) graphblas.Vector[T] {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("PageRank requires a square matrix, found %+v x %+v", n, a.Columns())
	}

	rank := graphblas.NewDenseVectorN[T](n)
	if n == 0 {
		return rank
	}

	// Total outgoing edge weight per vertex; zero marks a dangling vertex.
	out := make([]T, n)
	for iterator := a.Enumerate(); iterator.HasNext(); {
		u, _, w := iterator.Next()
		out[u] += w
	}

	// Each iteration pulls rank in along incoming edges, so multiply by
	// the transpose.
	at := graphblas.TransposeToCSR(ctx, a)

	inv := T(1) / T(n)
	for i := 0; i < n; i++ {
		rank.SetVec(i, inv)
	}

	weighted := graphblas.NewDenseVectorN[T](n)
	next := graphblas.NewDenseVectorN[T](n)

	for iteration := 0; iteration < maxIterations; iteration++ {
		select {
		case <-ctx.Done():
			return rank
		default:
		}

		dangling := T(0)
		for j := 0; j < n; j++ {
			r := rank.AtVec(j)
			if out[j] != 0 {
				weighted.SetVec(j, r/out[j])
			} else {
				weighted.SetVec(j, 0)
				dangling += r
			}
		}

		graphblas.MatrixVectorMultiply[T](ctx, at, weighted, nil, next)

		teleport := (1-damping)*inv + damping*dangling*inv

		delta := T(0)
		for i := 0; i < n; i++ {
			v := damping*next.AtVec(i) + teleport
			d := v - rank.AtVec(i)
			if d < 0 {
				d = -d
			}
			delta += d
			rank.SetVec(i, v)
		}

		if delta <= tolerance {
			break
		}
	}

	return rank
}
