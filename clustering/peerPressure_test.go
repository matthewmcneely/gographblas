// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering_test

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/clustering"
)

// referencePeerPressure tallies votes vertex by vertex over dense arrays,
// independent of the library's mxm formulation.
func referencePeerPressure(adj [][]float64, iterations int) []int {
	n := len(adj)
	share := make([][]float64, n)
	for u := range share {
		share[u] = make([]float64, n)
		heaviest, total := 0.0, 0.0
		for v, w := range adj[u] {
			share[u][v] = w
			heaviest = max(heaviest, w)
			total += w
		}
		if heaviest == 0 {
			heaviest = 1
		}
		share[u][u] += heaviest
		total += heaviest
		for v := range share[u] {
			share[u][v] /= total
		}
	}

	labels := make([]int, n)
	for v := range labels {
		labels[v] = v
	}

	for i := 0; i < iterations; i++ {
		next := make([]int, n)
		moved := false
		for v := 0; v < n; v++ {
			tally := make([]float64, n)
			for u := 0; u < n; u++ {
				tally[labels[u]] += share[u][v]
			}
			best := 0
			for c := range tally {
				if tally[c] > tally[best] {
					best = c
				}
			}
			next[v] = best
			moved = moved || best != labels[v]
		}
		labels = next
		if !moved {
			break
		}
	}

	renumber := map[int]int{}
	for v, l := range labels {
		if _, ok := renumber[l]; !ok {
			renumber[l] = len(renumber)
		}
		labels[v] = renumber[l]
	}
	return labels
}

func TestPeerPressureCliqueRing(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 1)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	assertLabels(t, clustering.PeerPressure[float64](context.Background(), g), want)
}

func TestPeerPressureFollowsEdgeDirection(t *testing.T) {
	// Two 4-cliques A (0-3) and B (4-7), plus vertex 8 that every member
	// of A points at and that points at every member of B. Votes travel
	// along edges, so 8 hears only from A and joins it; reversing every
	// edge sends it to B.
	edges, _, _ := cliqueRing(2, 4, 0)
	edges = undirected(edges[:len(edges)-2]) // drop the ring's bridges
	for i := 0; i < 4; i++ {
		edges = append(edges,
			graphblas.Edge[float64]{From: i, To: 8, Weight: 1},
			graphblas.Edge[float64]{From: 8, To: 4 + i, Weight: 1})
	}
	g := graphblas.NewCSRMatrixFromEdges(9, 9, edges)

	assertLabels(t, clustering.PeerPressure[float64](context.Background(), g), []int{0, 0, 0, 0, 1, 1, 1, 1, 0})

	reversed := graphblas.TransposeToCSR(context.Background(), g)
	assertLabels(t, clustering.PeerPressure[float64](context.Background(), reversed), []int{0, 0, 0, 0, 1, 1, 1, 1, 1})
}

func TestPeerPressureMatchesDenseReference(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	const n = 40

	adj := make([][]float64, n)
	var edges []graphblas.Edge[float64]
	for u := range adj {
		adj[u] = make([]float64, n)
		for v := range adj[u] {
			if u != v && rnd.Float64() < 0.1 {
				adj[u][v] = float64(1 + rnd.Intn(3))
				edges = append(edges, graphblas.Edge[float64]{From: u, To: v, Weight: adj[u][v]})
			}
		}
	}
	g := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	got := clustering.PeerPressure[float64](context.Background(), g)
	want := referencePeerPressure(adj, clustering.DefaultPeerPressureIterations)
	assertLabels(t, got, want)
}

func TestPeerPressurePlantedPartition(t *testing.T) {
	// Dense communities: at pIn 0.5 a seed in three can leave one split.
	g, want := planted(11, 4, 25, 0.8, 0.01)

	assertLabels(t, clustering.PeerPressure[float64](context.Background(), g), want)
}

func TestPeerPressureFloat32(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 1)
	g := graphblas.NewCSRMatrixFromEdges(n, n, to32(undirected(edges)))

	assertLabels(t, clustering.PeerPressure[float32](context.Background(), g), want)
}

func TestPeerPressureZeroIterations(t *testing.T) {
	edges, n, _ := cliqueRing(2, 3, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	assertLabels(t, clustering.PeerPressureWithOptions[float64](context.Background(), g, 0), []int{0, 1, 2, 3, 4, 5})
}

func TestPeerPressurePanics(t *testing.T) {
	assertPanics(t, "non-square", func() {
		clustering.PeerPressure[float64](context.Background(), graphblas.NewCSRMatrix[float64](2, 3))
	})
}
