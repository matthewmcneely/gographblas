// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering_test

import (
	"context"
	"math/rand"
	"slices"
	"testing"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/clustering"
)

// conductance is the weight of edges leaving set over the smaller of the
// set's volume and the rest of the graph's, computed from scratch.
func conductance(g graphblas.Matrix[float64], set []int) float64 {
	in := make(map[int]bool)
	for _, v := range set {
		in[v] = true
	}

	var cut, volume, total float64
	for iterator := g.Enumerate(); iterator.HasNext(); {
		u, v, w := iterator.Next()
		total += w
		if in[u] {
			volume += w
			if !in[v] {
				cut += w
			}
		}
	}
	return cut / min(volume, total-volume)
}

func TestLocalFindsSeedClique(t *testing.T) {
	// Two 5-cliques and an isolated vertex: each clique is the best cut
	// around any of its vertices.
	edges, n, want := cliqueRing(2, 5, 1)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))
	groups := clustering.Groups(want)

	for seed := 0; seed < n; seed++ {
		got := clustering.Local[float64](context.Background(), g, seed)
		if !slices.Equal(got, groups[want[seed]]) {
			t.Fatalf("Local(%d) = %v, want %v", seed, got, groups[want[seed]])
		}
	}
}

func TestLocalPrefersLowerConductance(t *testing.T) {
	// On a ring of four cliques, two adjacent cliques cut as few edges as
	// one clique does with twice the volume, so the sweep must do at least
	// as well as the seed's clique alone.
	edges, n, want := cliqueRing(4, 5, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))
	groups := clustering.Groups(want)

	for _, seed := range []int{0, 7, 12, 19} {
		got := clustering.Local[float64](context.Background(), g, seed)
		if !slices.Contains(got, seed) {
			t.Fatalf("Local(%d) = %v, want the seed included", seed, got)
		}
		if c, clique := conductance(g, got), conductance(g, groups[want[seed]]); c > clique {
			t.Fatalf("Local(%d) = %v with conductance %v, worse than its clique's %v", seed, got, c, clique)
		}
	}
}

func TestLocalTreatsGraphAsUndirected(t *testing.T) {
	edges, n, _ := cliqueRing(3, 4, 0)
	oneWay := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	if got := clustering.Local[float64](context.Background(), oneWay, 5); !slices.Equal(got, []int{4, 5, 6, 7}) {
		t.Fatalf("Local = %v, want [4 5 6 7]", got)
	}
}

func TestLocalKeepsCommunitiesWhole(t *testing.T) {
	// Conductance favors larger sets up to half the graph's volume, so the
	// cluster may merge whole blocks; it must not split one. A sweep cut
	// is approximate, so allow 10% slack either way.
	g, want := planted(19, 4, 25, 0.4, 0.01)
	groups := clustering.Groups(want)

	for seed := 0; seed < g.Rows(); seed += 7 {
		got := clustering.Local[float64](context.Background(), g, seed)

		inside := make([]int, len(groups))
		for _, v := range got {
			inside[want[v]]++
		}

		for b, block := range groups {
			share := float64(inside[b]) / float64(len(block))
			if (b == want[seed] && share < 0.9) || (share > 0.1 && share < 0.9) {
				t.Fatalf("Local(%d) = %v holds %d of block %d's %d vertices", seed, got, inside[b], b, len(block))
			}
		}
	}
}

func TestLocalContainsSeed(t *testing.T) {
	// Exact personalized PageRank per unit of degree peaks at the seed, so
	// it leads the sweep; the contract holds under the push's
	// approximation too. Random small graphs, every seed.
	for trial := int64(0); trial < 200; trial++ {
		rnd := rand.New(rand.NewSource(trial))
		n := 5 + rnd.Intn(10)

		var edges []graphblas.Edge[float64]
		for u := 0; u < n; u++ {
			for v := u + 1; v < n; v++ {
				if rnd.Float64() < 0.3 {
					edges = append(edges, graphblas.Edge[float64]{From: u, To: v, Weight: 1 + rnd.Float64()*4})
				}
			}
		}
		g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

		for seed := 0; seed < n; seed++ {
			if got := clustering.Local[float64](context.Background(), g, seed); !slices.Contains(got, seed) {
				t.Fatalf("trial %d: Local(%d) = %v, want the seed included", trial, seed, got)
			}
		}
	}
}

func TestLocalFloat32(t *testing.T) {
	edges, n, _ := cliqueRing(4, 5, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, to32(undirected(edges)))

	if got := clustering.Local[float32](context.Background(), g, 12); !slices.Equal(got, []int{10, 11, 12, 13, 14}) {
		t.Fatalf("Local = %v, want [10 11 12 13 14]", got)
	}
}

func TestLocalHugeEpsilonReturnsSeed(t *testing.T) {
	edges, n, _ := cliqueRing(2, 4, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	if got := clustering.LocalWithOptions[float64](context.Background(), g, 2, 0.15, 10); !slices.Equal(got, []int{2}) {
		t.Fatalf("Local = %v, want [2]", got)
	}
}

func TestLocalPanics(t *testing.T) {
	g := graphblas.NewCSRMatrix[float64](3, 3)

	assertPanics(t, "seed -1", func() { clustering.Local[float64](context.Background(), g, -1) })
	assertPanics(t, "seed 3", func() { clustering.Local[float64](context.Background(), g, 3) })
	assertPanics(t, "teleport 0", func() { clustering.LocalWithOptions[float64](context.Background(), g, 0, 0, 1e-6) })
	assertPanics(t, "teleport 1", func() { clustering.LocalWithOptions[float64](context.Background(), g, 0, 1, 1e-6) })
	assertPanics(t, "epsilon 0", func() { clustering.LocalWithOptions[float64](context.Background(), g, 0, 0.15, 0) })
	assertPanics(t, "non-square", func() {
		clustering.Local[float64](context.Background(), graphblas.NewCSRMatrix[float64](2, 3), 0)
	})
}
