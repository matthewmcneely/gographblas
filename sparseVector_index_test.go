// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas_test

import (
	"context"
	"testing"

	"github.com/rossmerr/graphblas"
)

// Regression tests for the SparseVector index lookup, which previously
// compared the element index against the number of stored entries. Any
// element whose index exceeded the entry count was unreadable, SetVec
// inserted duplicates, and sparse multiplies silently dropped terms.

func TestSparseVectorScatteredSetGet(t *testing.T) {
	v := graphblas.NewSparseVector[float64](5)
	v.SetVec(4, 1)
	v.SetVec(0, 2)
	v.SetVec(2, 3)

	want := []float64{2, 0, 3, 0, 1}
	for i, w := range want {
		if got := v.AtVec(i); got != w {
			t.Fatalf("AtVec(%d) = %v, want %v", i, got, w)
		}
	}

	if v.Values() != 3 {
		t.Fatalf("Values() = %d, want 3", v.Values())
	}
}

func TestSparseVectorSingleHighIndex(t *testing.T) {
	v := graphblas.NewSparseVector[float64](5)
	v.SetVec(3, 7)

	if got := v.AtVec(3); got != 7 {
		t.Fatalf("AtVec(3) = %v, want 7", got)
	}
}

func TestSparseVectorFromSparseArray(t *testing.T) {
	v := graphblas.NewSparseVectorFromArray([]float64{0, 0, 0, 5})

	if got := v.AtVec(3); got != 5 {
		t.Fatalf("AtVec(3) = %v, want 5", got)
	}
}

func TestSparseVectorOverwriteAfterScatteredSets(t *testing.T) {
	v := graphblas.NewSparseVector[float64](6)
	v.SetVec(5, 1)
	v.SetVec(1, 2)
	v.SetVec(5, 9) // must overwrite, not insert a duplicate

	if got := v.AtVec(5); got != 9 {
		t.Fatalf("AtVec(5) = %v, want 9", got)
	}
	if v.Values() != 2 {
		t.Fatalf("Values() = %d, want 2", v.Values())
	}

	v.SetVec(5, 0) // removing the entry must leave index 1 intact
	if got := v.AtVec(5); got != 0 {
		t.Fatalf("AtVec(5) = %v, want 0 after removal", got)
	}
	if got := v.AtVec(1); got != 2 {
		t.Fatalf("AtVec(1) = %v, want 2", got)
	}
}

// TestCSRMultiplyMatchesDense multiplies a genuinely sparse matrix in CSR
// form against the dense result. Before the fix, RowsAt and ColumnsAt
// returned SparseVectors whose high-index entries read as zero, so the
// products disagreed.
func TestCSRMultiplyMatchesDense(t *testing.T) {
	data := [][]float64{
		{0, 1, 1, 0, 0},
		{0, 0, 2, 1, 0},
		{1, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0},
	}

	dense := graphblas.NewDenseMatrixFromArrayN(data)
	csr := graphblas.NewCSRMatrixFromArray(data)

	want := dense.Multiply(dense)
	got := csr.Multiply(dense)

	if !got.Equal(want) {
		t.Fatalf("CSR multiply disagrees with dense multiply")
	}
}

// TestCSRMatrixVectorMultiply checks a sparse mxv against hand-computed
// values, including a row whose only entry sits at a high column index.
func TestCSRMatrixVectorMultiply(t *testing.T) {
	data := [][]float64{
		{0, 0, 1, 0, 0},
		{1, 0, 0, 0, 0},
		{1, 2, 0, 0, 1},
		{0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0},
	}

	m := graphblas.NewCSRMatrixFromArray(data)

	v := graphblas.NewDenseVectorN[float64](5)
	for i := 0; i < 5; i++ {
		v.SetVec(i, float64(i+1))
	}

	result := graphblas.NewDenseVectorN[float64](5)
	graphblas.MatrixVectorMultiply[float64](context.Background(), m, v, nil, result)

	want := []float64{3, 1, 10, 2, 0}
	for i, w := range want {
		if got := result.AtVec(i); got != w {
			t.Fatalf("result[%d] = %v, want %v", i, got, w)
		}
	}
}
