// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package centrality_test

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/centrality"
)

// referenceCloseness computes the Wasserman-Faust closeness from
// Floyd-Warshall distances, independent of the library's semiring path.
func referenceCloseness(adj [][]float64) []float64 {
	n := len(adj)
	inf := math.Inf(1)

	dist := make([][]float64, n)
	for i := range dist {
		dist[i] = make([]float64, n)
		for j := range dist[i] {
			switch {
			case i == j:
				dist[i][j] = 0
			case adj[i][j] != 0:
				dist[i][j] = adj[i][j]
			default:
				dist[i][j] = inf
			}
		}
	}

	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if d := dist[i][k] + dist[k][j]; d < dist[i][j] {
					dist[i][j] = d
				}
			}
		}
	}

	scores := make([]float64, n)
	for s := 0; s < n; s++ {
		sum := 0.0
		reach := 0
		for i := 0; i < n; i++ {
			if i != s && dist[s][i] < inf {
				sum += dist[s][i]
				reach++
			}
		}
		if reach > 0 && sum > 0 {
			r := float64(reach)
			scores[s] = (r / sum) * (r / float64(n-1))
		}
	}

	return scores
}

func TestClosenessChain(t *testing.T) {
	// 0 -> 1 -> 2 with unit weights: 0 reaches both (sum 3), 1 reaches one
	// (sum 1), 2 reaches nothing.
	m := graphblas.NewCSRMatrixFromEdges(3, 3, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 1},
		{From: 1, To: 2, Weight: 1},
	})

	c := centrality.Closeness[float64](context.Background(), m)

	want := []float64{(2.0 / 3.0) * (2.0 / 2.0), (1.0 / 1.0) * (1.0 / 2.0), 0}
	for i, w := range want {
		if math.Abs(c.AtVec(i)-w) > 1e-12 {
			t.Fatalf("closeness[%d] = %v, want %v", i, c.AtVec(i), w)
		}
	}
}

func TestClosenessWeighted(t *testing.T) {
	// The cheap route 0 -> 2 goes through 1 (cost 3), beating the direct
	// edge (cost 5).
	m := graphblas.NewCSRMatrixFromEdges(3, 3, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 2},
		{From: 0, To: 2, Weight: 5},
		{From: 1, To: 2, Weight: 1},
	})

	c := centrality.Closeness[float64](context.Background(), m)

	if got, want := c.AtVec(0), (2.0/5.0)*(2.0/2.0); math.Abs(got-want) > 1e-12 {
		t.Fatalf("closeness[0] = %v, want %v", got, want)
	}
	if got, want := c.AtVec(1), (1.0/1.0)*(1.0/2.0); math.Abs(got-want) > 1e-12 {
		t.Fatalf("closeness[1] = %v, want %v", got, want)
	}
	if got := c.AtVec(2); got != 0 {
		t.Fatalf("closeness[2] = %v, want 0", got)
	}
}

func TestClosenessMatchesReference(t *testing.T) {
	const n = 40
	rnd := rand.New(rand.NewSource(23))

	adj := make([][]float64, n)
	for r := range adj {
		adj[r] = make([]float64, n)
		for k := 0; k < 4; k++ {
			c := rnd.Intn(n)
			if c != r {
				adj[r][c] = rnd.Float64()*9 + 1
			}
		}
	}

	want := referenceCloseness(adj)

	matrices := map[string]graphblas.Matrix[float64]{
		"dense": graphblas.NewDenseMatrixFromArrayN(adj),
		"csr":   graphblas.NewCSRMatrixFromArray(adj),
	}

	for name, m := range matrices {
		c := centrality.Closeness[float64](context.Background(), m)

		for i, w := range want {
			if math.Abs(c.AtVec(i)-w) > 1e-9 {
				t.Fatalf("%s: closeness[%d] = %v, want %v", name, i, c.AtVec(i), w)
			}
		}
	}
}

func BenchmarkCloseness_1000(b *testing.B) {
	const n = 1000
	rnd := rand.New(rand.NewSource(24))
	edges := make([]graphblas.Edge[float64], 0, n*8)
	for r := 0; r < n; r++ {
		for k := 0; k < 8; k++ {
			edges = append(edges, graphblas.Edge[float64]{From: r, To: rnd.Intn(n), Weight: rnd.Float64() + 0.5})
		}
	}
	m := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		centrality.Closeness[float64](context.Background(), m)
	}
}
