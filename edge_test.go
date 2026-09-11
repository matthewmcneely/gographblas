// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas_test

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rossmerr/graphblas"
)

func TestNewCSRMatrixFromEdgesMatchesFromArray(t *testing.T) {
	const n = 100
	rnd := rand.New(rand.NewSource(13))

	data := make([][]float64, n)
	for r := range data {
		data[r] = make([]float64, n)
		for k := 0; k < 8; k++ {
			data[r][rnd.Intn(n)] = rnd.Float64() + 0.5
		}
	}

	var edges []graphblas.Edge[float64]
	for r := range data {
		for c, v := range data[r] {
			if v != 0 {
				edges = append(edges, graphblas.Edge[float64]{From: r, To: c, Weight: v})
			}
		}
	}

	// Shuffle so the constructor sees edges in arbitrary order.
	rnd.Shuffle(len(edges), func(i, j int) { edges[i], edges[j] = edges[j], edges[i] })

	want := graphblas.NewCSRMatrixFromArray(data)
	got := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	if !got.Equal(want) {
		t.Fatal("edge-built matrix disagrees with array-built matrix")
	}
}

func TestNewCSRMatrixFromEdgesDuplicatesSum(t *testing.T) {
	edges := []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 2},
		{From: 0, To: 1, Weight: 3},
		{From: 1, To: 0, Weight: 5},
		{From: 1, To: 0, Weight: -5}, // sums to zero, must not be stored
		{From: 1, To: 2, Weight: 0},  // zero weight, must not be stored
	}

	m := graphblas.NewCSRMatrixFromEdges(2, 3, edges)

	if got := m.At(0, 1); got != 5 {
		t.Fatalf("At(0,1) = %v, want 5", got)
	}
	if got := m.At(1, 0); got != 0 {
		t.Fatalf("At(1,0) = %v, want 0", got)
	}
	if m.Values() != 1 {
		t.Fatalf("Values() = %d, want 1", m.Values())
	}
}

func TestNewCSRMatrixFromEdgesEmpty(t *testing.T) {
	m := graphblas.NewCSRMatrixFromEdges[float64](3, 3, nil)

	if m.Values() != 0 {
		t.Fatalf("Values() = %d, want 0", m.Values())
	}
	if got := m.At(1, 1); got != 0 {
		t.Fatalf("At(1,1) = %v, want 0", got)
	}
}

// TestTransposeToCSRMatchesGeneric compares the transpose against the
// masked generic Transpose applied to the same input.
func TestTransposeToCSRMatchesGeneric(t *testing.T) {
	const n = 60
	_, csr, _ := randomSparse(14, n, 5)

	fast := graphblas.TransposeToCSR[float64](context.Background(), csr)

	generic := graphblas.NewCSRMatrix[float64](n, n)
	graphblas.Transpose[float64](context.Background(), csr, nil, generic)

	if !fast.Equal(generic) {
		t.Fatal("TransposeToCSR disagrees with generic Transpose")
	}
}

func benchmarkEdges(n, perRow int) []graphblas.Edge[float64] {
	rnd := rand.New(rand.NewSource(15))
	edges := make([]graphblas.Edge[float64], 0, n*perRow)
	for r := 0; r < n; r++ {
		for k := 0; k < perRow; k++ {
			edges = append(edges, graphblas.Edge[float64]{From: r, To: rnd.Intn(n), Weight: rnd.Float64() + 0.5})
		}
	}
	return edges
}

func BenchmarkNewCSRMatrixFromEdges_10000(b *testing.B) {
	edges := benchmarkEdges(10000, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graphblas.NewCSRMatrixFromEdges(10000, 10000, edges)
	}
}

func BenchmarkTransposeToCSR_10000(b *testing.B) {
	m := graphblas.NewCSRMatrixFromEdges(10000, 10000, benchmarkEdges(10000, 10))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graphblas.TransposeToCSR[float64](context.Background(), m)
	}
}
