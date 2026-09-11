// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package centrality_test

import (
	"context"
	"math"
	"testing"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/centrality"
)

// referencePageRank is an independent straightforward power iteration used
// to validate the library implementation.
func referencePageRank(adj [][]float64, damping, tolerance float64, maxIterations int) []float64 {
	n := len(adj)
	out := make([]float64, n)
	for u := range adj {
		for _, w := range adj[u] {
			out[u] += w
		}
	}

	rank := make([]float64, n)
	for i := range rank {
		rank[i] = 1 / float64(n)
	}

	for iteration := 0; iteration < maxIterations; iteration++ {
		next := make([]float64, n)
		dangling := 0.0
		for u := 0; u < n; u++ {
			if out[u] == 0 {
				dangling += rank[u]
				continue
			}
			for v := 0; v < n; v++ {
				if adj[u][v] != 0 {
					next[v] += rank[u] * adj[u][v] / out[u]
				}
			}
		}

		teleport := (1-damping)/float64(n) + damping*dangling/float64(n)
		delta := 0.0
		for i := 0; i < n; i++ {
			value := damping*next[i] + teleport
			delta += math.Abs(value - rank[i])
			rank[i] = value
		}

		if delta <= tolerance {
			break
		}
	}

	return rank
}

// testGraph has weighted edges, a dangling vertex (3) and a vertex with no
// in-links (4).
var testGraph = [][]float64{
	{0, 1, 1, 0, 0},
	{0, 0, 2, 1, 0},
	{1, 0, 0, 0, 0},
	{0, 0, 0, 0, 0},
	{0, 0, 1, 0, 0},
}

func TestPageRankMatchesReference(t *testing.T) {
	want := referencePageRank(testGraph, centrality.DefaultDamping, centrality.DefaultTolerance, centrality.DefaultIterations)

	matrices := map[string]graphblas.Matrix[float64]{
		"dense": graphblas.NewDenseMatrixFromArrayN(testGraph),
		"csr":   graphblas.NewCSRMatrixFromArray(testGraph),
	}

	for name, m := range matrices {
		rank := centrality.PageRank[float64](context.Background(), m)

		if rank.Length() != len(want) {
			t.Fatalf("%s: rank length %d, want %d", name, rank.Length(), len(want))
		}

		sum := 0.0
		for i, w := range want {
			got := rank.AtVec(i)
			if math.Abs(got-w) > 1e-9 {
				t.Fatalf("%s: rank[%d] = %v, want %v", name, i, got, w)
			}
			sum += got
		}

		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("%s: ranks sum to %v, want 1", name, sum)
		}
	}
}

func TestPageRankTwoVertexCycle(t *testing.T) {
	m := graphblas.NewDenseMatrixFromArrayN([][]float64{
		{0, 1},
		{1, 0},
	})

	rank := centrality.PageRank[float64](context.Background(), m)

	for i := 0; i < 2; i++ {
		if math.Abs(rank.AtVec(i)-0.5) > 1e-6 {
			t.Fatalf("rank[%d] = %v, want 0.5", i, rank.AtVec(i))
		}
	}
}

func TestPageRankStarCenterRanksHighest(t *testing.T) {
	// Vertices 1..3 all link to vertex 0.
	m := graphblas.NewDenseMatrixFromArrayN([][]float64{
		{0, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{1, 0, 0, 0},
	})

	rank := centrality.PageRank[float64](context.Background(), m)

	center := rank.AtVec(0)
	for i := 1; i < 4; i++ {
		if center <= rank.AtVec(i) {
			t.Fatalf("rank[0] = %v not greater than rank[%d] = %v", center, i, rank.AtVec(i))
		}
	}
}

func TestPageRankWithOptionsFloat32(t *testing.T) {
	m := graphblas.NewDenseMatrixFromArrayN([][]float32{
		{0, 1},
		{1, 0},
	})

	rank := centrality.PageRankWithOptions[float32](context.Background(), m, 0.85, 1e-5, 100)

	for i := 0; i < 2; i++ {
		if math.Abs(float64(rank.AtVec(i))-0.5) > 1e-4 {
			t.Fatalf("rank[%d] = %v, want 0.5", i, rank.AtVec(i))
		}
	}
}
