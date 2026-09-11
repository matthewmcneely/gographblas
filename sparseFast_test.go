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

// randomSparse builds an n x n matrix with roughly nnzPerRow stored
// entries per row, returned as the raw array plus CSR and CSC forms.
func randomSparse(seed int64, n, nnzPerRow int) ([][]float64, *graphblas.CSRMatrix[float64], *graphblas.CSCMatrix[float64]) {
	rnd := rand.New(rand.NewSource(seed))

	data := make([][]float64, n)
	for r := range data {
		data[r] = make([]float64, n)
		for k := 0; k < nnzPerRow; k++ {
			data[r][rnd.Intn(n)] = rnd.Float64() + 0.5
		}
	}

	return data, graphblas.NewCSRMatrixFromArray(data), graphblas.NewCSCMatrixFromArray(data)
}

func TestSparseMatrixVectorMatchesReference(t *testing.T) {
	const n = 200
	data, csr, csc := randomSparse(9, n, 10)

	rnd := rand.New(rand.NewSource(10))
	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, rnd.NormFloat64())
	}

	want := make([]float64, n)
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			want[r] += data[r][c] * x.AtVec(c)
		}
	}

	matrices := map[string]graphblas.Matrix[float64]{"csr": csr, "csc": csc}
	for name, m := range matrices {
		result := graphblas.NewDenseVectorN[float64](n)
		graphblas.MatrixVectorMultiply[float64](context.Background(), m, x, nil, result)

		for r := 0; r < n; r++ {
			got := result.AtVec(r)
			diff := got - want[r]
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-9 {
				t.Fatalf("%s: result[%d] = %v, want %v", name, r, got, want[r])
			}
		}
	}
}

// TestSparseMatrixVectorMasked checks that masked output rows keep their
// previous contents, matching the generic path's semantics.
func TestSparseMatrixVectorMasked(t *testing.T) {
	const n = 64
	_, csr, csc := randomSparse(11, n, 6)

	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, 1)
	}

	// The mask skips even rows: DenseVectorNumber.Element(r, c) is true
	// when the value is positive, and true means the cell is not written.
	mask := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i += 2 {
		mask.SetVec(i, 1)
	}

	matrices := map[string]graphblas.Matrix[float64]{"csr": csr, "csc": csc}
	for name, m := range matrices {
		expected := graphblas.NewDenseVectorN[float64](n)
		graphblas.MatrixVectorMultiply[float64](context.Background(), m, x, nil, expected)

		const sentinel = -12345.0
		result := graphblas.NewDenseVectorN[float64](n)
		for i := 0; i < n; i++ {
			result.SetVec(i, sentinel)
		}

		graphblas.MatrixVectorMultiply[float64](context.Background(), m, x, mask, result)

		for r := 0; r < n; r++ {
			got := result.AtVec(r)
			if r%2 == 0 {
				if got != sentinel {
					t.Fatalf("%s: masked row %d was written: %v", name, r, got)
				}
			} else if got != expected.AtVec(r) {
				t.Fatalf("%s: result[%d] = %v, want %v", name, r, got, expected.AtVec(r))
			}
		}
	}
}

func TestSparseMatrixVectorInt(t *testing.T) {
	data := [][]int{
		{0, 2, 0},
		{0, 0, 3},
		{4, 0, 0},
	}

	csr := graphblas.NewCSRMatrixFromArray(data)

	x := graphblas.NewDenseVectorN[int](3)
	for i := 0; i < 3; i++ {
		x.SetVec(i, i+1)
	}

	result := graphblas.NewDenseVectorN[int](3)
	graphblas.MatrixVectorMultiply[int](context.Background(), csr, x, nil, result)

	want := []int{4, 9, 4}
	for i, w := range want {
		if result.AtVec(i) != w {
			t.Fatalf("result[%d] = %v, want %v", i, result.AtVec(i), w)
		}
	}
}

func benchmarkSparseMxv(b *testing.B, build func([][]float64) graphblas.Matrix[float64]) {
	const n = 10000
	data, _, _ := randomSparse(12, n, 10)
	m := build(data)

	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, float64(i%7)+1)
	}
	result := graphblas.NewDenseVectorN[float64](n)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graphblas.MatrixVectorMultiply[float64](context.Background(), m, x, nil, result)
	}
}

func BenchmarkCSRMatrixVector_10000(b *testing.B) {
	benchmarkSparseMxv(b, func(data [][]float64) graphblas.Matrix[float64] {
		return graphblas.NewCSRMatrixFromArray(data)
	})
}

func BenchmarkCSCMatrixVector_10000(b *testing.B) {
	benchmarkSparseMxv(b, func(data [][]float64) graphblas.Matrix[float64] {
		return graphblas.NewCSCMatrixFromArray(data)
	})
}
