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

// denseProduct multiplies two raw arrays the textbook way.
func denseProduct(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for r := range a {
		out[r] = make([]float64, len(b[0]))
		for k := range b {
			if a[r][k] == 0 {
				continue
			}
			for c := range b[k] {
				out[r][c] += a[r][k] * b[k][c]
			}
		}
	}
	return out
}

func TestSparseMatrixMatrixMatchesReference(t *testing.T) {
	const n = 150
	aData, a, _ := randomSparse(21, n, 6)
	bData, b, _ := randomSparse(22, n, 6)
	want := denseProduct(aData, bData)

	result := graphblas.NewCSRMatrix[float64](n, n)
	graphblas.MatrixMatrixMultiply[float64](context.Background(), a, b, nil, result)

	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			diff := result.At(r, c) - want[r][c]
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-9 {
				t.Fatalf("result[%d][%d] = %v, want %v", r, c, result.At(r, c), want[r][c])
			}
		}
	}

	stored := 0
	for iterator := result.Enumerate(); iterator.HasNext(); {
		iterator.Next()
		stored++
	}
	if stored != result.Values() {
		t.Fatalf("enumerated %d entries, Values reports %d", stored, result.Values())
	}
}

func TestSparseMatrixMatrixRectangular(t *testing.T) {
	a := graphblas.NewCSRMatrixFromArray([][]float64{
		{1, 0, 2},
		{0, 3, 0},
	})
	b := graphblas.NewCSRMatrixFromArray([][]float64{
		{1, 0, 0, 4},
		{0, 0, 5, 0},
		{0, 6, 0, 1},
	})
	want := [][]float64{
		{1, 12, 0, 6},
		{0, 0, 15, 0},
	}

	result := graphblas.NewCSRMatrix[float64](2, 4)
	graphblas.MatrixMatrixMultiply[float64](context.Background(), a, b, nil, result)

	for r := range want {
		for c := range want[r] {
			if result.At(r, c) != want[r][c] {
				t.Fatalf("result[%d][%d] = %v, want %v", r, c, result.At(r, c), want[r][c])
			}
		}
	}
}

func TestSparseMatrixMatrixDropsCancelledSums(t *testing.T) {
	// Row 0 of the product is 1*1 + 1*(-1) = 0 in column 0, which must
	// not be stored.
	a := graphblas.NewCSRMatrixFromArray([][]int{
		{1, 1},
		{0, 2},
	})
	b := graphblas.NewCSRMatrixFromArray([][]int{
		{1, 3},
		{-1, 0},
	})

	result := graphblas.NewCSRMatrix[int](2, 2)
	graphblas.MatrixMatrixMultiply[int](context.Background(), a, b, nil, result)

	want := [][]int{{0, 3}, {-2, 0}}
	for r := range want {
		for c := range want[r] {
			if result.At(r, c) != want[r][c] {
				t.Fatalf("result[%d][%d] = %v, want %v", r, c, result.At(r, c), want[r][c])
			}
		}
	}

	if result.Values() != 2 {
		t.Fatalf("stored %d entries, want 2", result.Values())
	}
}

func TestSparseMatrixMatrixReplacesOutput(t *testing.T) {
	a := graphblas.NewCSRMatrixFromArray([][]float64{{0, 1}, {1, 0}})

	// The output starts populated; the product must replace it.
	result := graphblas.NewCSRMatrixFromArray([][]float64{{7, 7}, {7, 7}})
	graphblas.MatrixMatrixMultiply[float64](context.Background(), a, a, nil, result)

	want := [][]float64{{1, 0}, {0, 1}}
	for r := range want {
		for c := range want[r] {
			if result.At(r, c) != want[r][c] {
				t.Fatalf("result[%d][%d] = %v, want %v", r, c, result.At(r, c), want[r][c])
			}
		}
	}
}

func TestSparseMatrixMatrixAliasedOutput(t *testing.T) {
	data := [][]float64{
		{0, 2, 0},
		{0, 0, 3},
		{1, 0, 0},
	}
	want := denseProduct(data, data)

	a := graphblas.NewCSRMatrixFromArray(data)
	graphblas.MatrixMatrixMultiply[float64](context.Background(), a, a, nil, a)

	for r := range want {
		for c := range want[r] {
			if a.At(r, c) != want[r][c] {
				t.Fatalf("result[%d][%d] = %v, want %v", r, c, a.At(r, c), want[r][c])
			}
		}
	}
}

func BenchmarkCSRMatrixMatrix_2000(b *testing.B) {
	const n = 2000
	_, a, _ := randomSparse(23, n, 10)
	result := graphblas.NewCSRMatrix[float64](n, n)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graphblas.MatrixMatrixMultiply[float64](context.Background(), a, a, nil, result)
	}
}
