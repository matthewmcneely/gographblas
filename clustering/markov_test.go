// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering_test

import (
	"context"
	"math"
	"testing"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/clustering"
)

// referenceMarkov runs MCL over dense arrays with the library's self-loop,
// pruning and convergence rules, independent of the sparse mxm path, and
// labels the components of the converged flow by their lowest vertex.
func referenceMarkov(adj [][]float64, inflation float64, iterations int) []int {
	n := len(adj)
	m := make([][]float64, n)
	for u := range m {
		m[u] = make([]float64, n)
		for v := range m[u] {
			m[u][v] = adj[u][v] + adj[v][u]
		}
		heaviest := 0.0
		for _, w := range m[u] {
			heaviest = math.Max(heaviest, w)
		}
		if heaviest == 0 {
			heaviest = 1
		}
		m[u][u] += heaviest
	}

	normalize := func(m [][]float64, power float64) {
		for u := range m {
			sum, largest := 0.0, 0.0
			for v := range m[u] {
				m[u][v] = math.Pow(m[u][v], power)
				sum += m[u][v]
				largest = math.Max(largest, m[u][v])
			}
			kept := 0.0
			for v := range m[u] {
				if m[u][v] != largest && m[u][v]/sum < 1e-5 {
					m[u][v] = 0
				}
				kept += m[u][v]
			}
			for v := range m[u] {
				m[u][v] /= kept
			}
		}
	}
	normalize(m, 1)

	for i := 0; i < iterations; i++ {
		next := make([][]float64, n)
		for u := range next {
			next[u] = make([]float64, n)
			for k := range m[u] {
				for v := range m[k] {
					next[u][v] += m[u][k] * m[k][v]
				}
			}
		}
		normalize(next, inflation)

		change := 0.0
		for u := range m {
			for v := range m[u] {
				change = math.Max(change, math.Abs(next[u][v]-m[u][v]))
			}
		}
		m = next
		if change < 1e-6 {
			break
		}
	}

	labels := make([]int, n)
	for u := range labels {
		labels[u] = u
	}
	for changed := true; changed; {
		changed = false
		for u := range m {
			for v := range m[u] {
				if m[u][v] > 0 && labels[u] != labels[v] {
					l := min(labels[u], labels[v])
					labels[u], labels[v] = l, l
					changed = true
				}
			}
		}
	}

	renumber := map[int]int{}
	for u, l := range labels {
		if _, ok := renumber[l]; !ok {
			renumber[l] = len(renumber)
		}
		labels[u] = renumber[l]
	}
	return labels
}

func TestMarkovCliqueRing(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 1)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	assertLabels(t, clustering.Markov[float64](context.Background(), g), want)
}

func TestMarkovTreatsGraphAsUndirected(t *testing.T) {
	edges, n, want := cliqueRing(3, 4, 0)
	oneWay := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	assertLabels(t, clustering.Markov[float64](context.Background(), oneWay), want)
}

func TestMarkovFloat32(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 1)
	g := graphblas.NewCSRMatrixFromEdges(n, n, to32(undirected(edges)))

	assertLabels(t, clustering.Markov[float32](context.Background(), g), want)
}

func TestMarkovMatchesDenseReference(t *testing.T) {
	for _, inflation := range []float64{1.4, 2, 3} {
		g, _ := planted(5, 3, 15, 0.4, 0.05)

		adj := make([][]float64, g.Rows())
		for u := range adj {
			adj[u] = make([]float64, g.Columns())
		}
		for iterator := g.Enumerate(); iterator.HasNext(); {
			u, v, w := iterator.Next()
			adj[u][v] = w
		}
		// The library symmetrizes a + a'; halve so the reference's
		// adj + adj' sees the same weights.
		for u := range adj {
			for v := range adj[u] {
				adj[u][v] /= 2
			}
		}

		got := clustering.MarkovWithOptions[float64](context.Background(), g, inflation, clustering.DefaultMarkovIterations)
		want := referenceMarkov(adj, inflation, clustering.DefaultMarkovIterations)
		assertLabels(t, got, want)
	}
}

func TestMarkovPlantedPartition(t *testing.T) {
	g, want := planted(7, 4, 25, 0.5, 0.01)

	assertLabels(t, clustering.Markov[float64](context.Background(), g), want)
}

func TestMarkovEmpty(t *testing.T) {
	g := graphblas.NewCSRMatrix[float64](0, 0)

	if got := clustering.Markov[float64](context.Background(), g); len(got) != 0 {
		t.Fatalf("labels = %v, want none", got)
	}
}

func TestMarkovCancelled(t *testing.T) {
	edges, n, _ := cliqueRing(4, 5, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if got := clustering.Markov[float64](ctx, g); len(got) != n {
		t.Fatalf("got %d labels, want %d", len(got), n)
	}
}

func TestMarkovPanics(t *testing.T) {
	assertPanics(t, "non-square", func() {
		clustering.Markov[float64](context.Background(), graphblas.NewCSRMatrix[float64](2, 3))
	})
	assertPanics(t, "inflation 1", func() {
		clustering.MarkovWithOptions[float64](context.Background(), graphblas.NewCSRMatrix[float64](2, 2), 1, 10)
	})
}
