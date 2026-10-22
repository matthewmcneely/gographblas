// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering_test

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/rossmerr/graphblas"
)

// undirected lists each edge in both directions.
func undirected(edges []graphblas.Edge[float64]) []graphblas.Edge[float64] {
	out := slices.Clone(edges)
	for _, e := range edges {
		out = append(out, graphblas.Edge[float64]{From: e.To, To: e.From, Weight: e.Weight})
	}
	return out
}

// cliqueRing builds k cliques of the given size, each joined to the next
// by a single edge, followed by isolated vertices. It returns the edges
// one way only, the vertex count, and the expected labels.
func cliqueRing(k, size, isolated int) ([]graphblas.Edge[float64], int, []int) {
	var edges []graphblas.Edge[float64]
	var want []int
	for c := 0; c < k; c++ {
		for i := 0; i < size; i++ {
			want = append(want, c)
			for j := i + 1; j < size; j++ {
				edges = append(edges, graphblas.Edge[float64]{From: c*size + i, To: c*size + j, Weight: 1})
			}
		}
		if k > 1 {
			edges = append(edges, graphblas.Edge[float64]{From: c*size + size - 1, To: ((c + 1) % k) * size, Weight: 1})
		}
	}
	for i := 0; i < isolated; i++ {
		want = append(want, k+i)
	}
	return edges, k*size + isolated, want
}

// planted builds an undirected random graph of equal blocks with edge
// probability pIn inside a block and pOut between blocks, returning the
// matrix and the block of each vertex.
func planted(seed int64, blocks, size int, pIn, pOut float64) (*graphblas.CSRMatrix[float64], []int) {
	rnd := rand.New(rand.NewSource(seed))
	n := blocks * size

	var edges []graphblas.Edge[float64]
	for u := 0; u < n; u++ {
		for v := u + 1; v < n; v++ {
			p := pOut
			if u/size == v/size {
				p = pIn
			}
			if rnd.Float64() < p {
				edges = append(edges, graphblas.Edge[float64]{From: u, To: v, Weight: 1})
			}
		}
	}

	want := make([]int, n)
	for v := range want {
		want[v] = v / size
	}
	return graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges)), want
}

func to32(edges []graphblas.Edge[float64]) []graphblas.Edge[float32] {
	out := make([]graphblas.Edge[float32], len(edges))
	for i, e := range edges {
		out[i] = graphblas.Edge[float32]{From: e.From, To: e.To, Weight: float32(e.Weight)}
	}
	return out
}

func assertLabels(t *testing.T, got, want []int) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("labels = %v, want %v", got, want)
	}
}

func assertPanics(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s did not panic", name)
		}
	}()
	f()
}
