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

// referenceBetweenness is the classic queue-based Brandes algorithm,
// independent of the library's matrix machinery, used to validate the
// mxv-based implementation.
func referenceBetweenness(adj [][]float64) []float64 {
	n := len(adj)
	bc := make([]float64, n)

	for s := 0; s < n; s++ {
		var stack []int
		preds := make([][]int, n)
		sigma := make([]float64, n)
		dist := make([]int, n)
		for i := range dist {
			dist[i] = -1
		}
		sigma[s] = 1
		dist[s] = 0

		queue := []int{s}
		for len(queue) > 0 {
			v := queue[0]
			queue = queue[1:]
			stack = append(stack, v)

			for w := 0; w < n; w++ {
				if adj[v][w] == 0 {
					continue
				}
				if dist[w] < 0 {
					dist[w] = dist[v] + 1
					queue = append(queue, w)
				}
				if dist[w] == dist[v]+1 {
					sigma[w] += sigma[v]
					preds[w] = append(preds[w], v)
				}
			}
		}

		delta := make([]float64, n)
		for i := len(stack) - 1; i >= 0; i-- {
			w := stack[i]
			for _, v := range preds[w] {
				delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
			}
			if w != s {
				bc[w] += delta[w]
			}
		}
	}

	return bc
}

func TestBetweennessDiamond(t *testing.T) {
	// Two equal shortest paths 0 -> 3, so vertices 1 and 2 each carry half
	// a pair.
	m := graphblas.NewCSRMatrixFromEdges(4, 4, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 1},
		{From: 0, To: 2, Weight: 1},
		{From: 1, To: 3, Weight: 1},
		{From: 2, To: 3, Weight: 1},
	})

	bc := centrality.Betweenness[float64](context.Background(), m)

	want := []float64{0, 0.5, 0.5, 0}
	for i, w := range want {
		if math.Abs(bc.AtVec(i)-w) > 1e-12 {
			t.Fatalf("bc[%d] = %v, want %v", i, bc.AtVec(i), w)
		}
	}
}

func TestBetweennessPath(t *testing.T) {
	// Directed path 0 -> 1 -> 2 -> 3: vertex 1 carries pairs (0,2) and
	// (0,3), vertex 2 carries (0,3) and (1,3).
	m := graphblas.NewCSRMatrixFromEdges(4, 4, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 1},
		{From: 1, To: 2, Weight: 1},
		{From: 2, To: 3, Weight: 1},
	})

	bc := centrality.Betweenness[float64](context.Background(), m)

	want := []float64{0, 2, 2, 0}
	for i, w := range want {
		if math.Abs(bc.AtVec(i)-w) > 1e-12 {
			t.Fatalf("bc[%d] = %v, want %v", i, bc.AtVec(i), w)
		}
	}
}

func TestBetweennessMatchesReference(t *testing.T) {
	const n = 40
	rnd := rand.New(rand.NewSource(21))

	adj := make([][]float64, n)
	for r := range adj {
		adj[r] = make([]float64, n)
		for k := 0; k < 4; k++ {
			c := rnd.Intn(n)
			if c != r {
				// Weights vary to prove they are ignored.
				adj[r][c] = rnd.Float64()*9 + 1
			}
		}
	}

	want := referenceBetweenness(adj)

	matrices := map[string]graphblas.Matrix[float64]{
		"dense": graphblas.NewDenseMatrixFromArrayN(adj),
		"csr":   graphblas.NewCSRMatrixFromArray(adj),
	}

	for name, m := range matrices {
		bc := centrality.Betweenness[float64](context.Background(), m)

		for i, w := range want {
			if math.Abs(bc.AtVec(i)-w) > 1e-9 {
				t.Fatalf("%s: bc[%d] = %v, want %v", name, i, bc.AtVec(i), w)
			}
		}
	}
}

func TestBetweennessFromSourcesSubset(t *testing.T) {
	// Restricting sources to {0} counts only pairs starting at 0: on the
	// path 0 -> 1 -> 2 -> 3, vertex 1 carries (0,2) and (0,3), vertex 2
	// carries (0,3).
	m := graphblas.NewCSRMatrixFromEdges(4, 4, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 1},
		{From: 1, To: 2, Weight: 1},
		{From: 2, To: 3, Weight: 1},
	})

	bc := centrality.BetweennessFromSources[float64](context.Background(), m, []int{0})

	want := []float64{0, 2, 1, 0}
	for i, w := range want {
		if math.Abs(bc.AtVec(i)-w) > 1e-12 {
			t.Fatalf("bc[%d] = %v, want %v", i, bc.AtVec(i), w)
		}
	}
}

func BenchmarkBetweenness_1000(b *testing.B) {
	const n = 1000
	rnd := rand.New(rand.NewSource(22))
	edges := make([]graphblas.Edge[float64], 0, n*8)
	for r := 0; r < n; r++ {
		for k := 0; k < 8; k++ {
			edges = append(edges, graphblas.Edge[float64]{From: r, To: rnd.Intn(n), Weight: 1})
		}
	}
	m := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		centrality.Betweenness[float64](context.Background(), m)
	}
}
