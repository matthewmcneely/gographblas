// Copyright (c) 2026 Matthew McNeely
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering_test

import (
	"context"
	"testing"

	"github.com/matthewmcneely/gographblas"
	"github.com/matthewmcneely/gographblas/clustering"
)

func TestSpectralCliqueRing(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	assertLabels(t, clustering.Spectral[float64](context.Background(), g, 4), want)
}

func TestSpectralIsolatedVertexIsItsOwnComponent(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 1)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	assertLabels(t, clustering.Spectral[float64](context.Background(), g, 5), want)

	// With two clusters, the components win over the cliques.
	twoWay := make([]int, n)
	twoWay[n-1] = 1
	assertLabels(t, clustering.Spectral[float64](context.Background(), g, 2), twoWay)
}

func TestSpectralTreatsGraphAsUndirected(t *testing.T) {
	edges, n, want := cliqueRing(3, 4, 0)
	oneWay := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	assertLabels(t, clustering.Spectral[float64](context.Background(), oneWay, 3), want)
}

func TestSpectralPlantedPartition(t *testing.T) {
	// Sparser than the peer pressure and Markov cases: the global
	// embedding still separates the blocks.
	g, want := planted(13, 3, 30, 0.3, 0.02)

	assertLabels(t, clustering.Spectral[float64](context.Background(), g, 3), want)
}

func TestSpectralFloat32(t *testing.T) {
	edges, n, want := cliqueRing(4, 5, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, to32(undirected(edges)))

	assertLabels(t, clustering.Spectral[float32](context.Background(), g, 4), want)
}

func TestSpectralDeterministic(t *testing.T) {
	g, _ := planted(17, 4, 20, 0.2, 0.05)

	first := clustering.Spectral[float64](context.Background(), g, 4)
	for i := 0; i < 3; i++ {
		assertLabels(t, clustering.Spectral[float64](context.Background(), g, 4), first)
	}
}

func TestSpectralSingleCluster(t *testing.T) {
	edges, n, _ := cliqueRing(2, 3, 0)
	g := graphblas.NewCSRMatrixFromEdges(n, n, undirected(edges))

	assertLabels(t, clustering.Spectral[float64](context.Background(), g, 1), make([]int, n))
}

func TestSpectralEmpty(t *testing.T) {
	g := graphblas.NewCSRMatrix[float64](0, 0)

	if got := clustering.Spectral[float64](context.Background(), g, 2); len(got) != 0 {
		t.Fatalf("labels = %v, want none", got)
	}
}

func TestSpectralPanics(t *testing.T) {
	g := graphblas.NewCSRMatrix[float64](3, 3)

	assertPanics(t, "k = 0", func() { clustering.Spectral[float64](context.Background(), g, 0) })
	assertPanics(t, "k > n", func() { clustering.Spectral[float64](context.Background(), g, 4) })
	assertPanics(t, "non-square", func() {
		clustering.Spectral[float64](context.Background(), graphblas.NewCSRMatrix[float64](2, 3), 1)
	})
}
