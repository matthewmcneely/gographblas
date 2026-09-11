// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas_test

import (
	"context"
	"math"
	"testing"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/binaryop"
)

// TestSemiringPlusTimesMatchesMultiply cross-validates the semiring
// multiply against the native plus-times multiply on a random sparse
// matrix.
func TestSemiringPlusTimesMatchesMultiply(t *testing.T) {
	const n = 200
	_, csr, csc := randomSparse(16, n, 10)

	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, float64(i%13)-6)
	}

	matrices := map[string]graphblas.Matrix[float64]{"csr": csr, "csc": csc}
	for name, m := range matrices {
		want := graphblas.NewDenseVectorN[float64](n)
		graphblas.MatrixVectorMultiply[float64](context.Background(), m, x, nil, want)

		got := graphblas.NewDenseVectorN[float64](n)
		graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), m, x, nil, got, binaryop.PlusTimes[float64]())

		for i := 0; i < n; i++ {
			diff := math.Abs(got.AtVec(i) - want.AtVec(i))
			if diff > 1e-9 {
				t.Fatalf("%s: got[%d] = %v, want %v", name, i, got.AtVec(i), want.AtVec(i))
			}
		}
	}
}

// TestSemiringMinPlus checks one shortest-path relaxation step by hand.
// With distances d = [0, +Inf, +Inf] and edges 0->1 (5), 0->2 (2),
// 1->2 (1), one (min, +) multiply by the transpose yields the one-hop
// distances [+Inf, 5, 2].
func TestSemiringMinPlus(t *testing.T) {
	edges := []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 5},
		{From: 0, To: 2, Weight: 2},
		{From: 1, To: 2, Weight: 1},
	}

	g := graphblas.NewCSRMatrixFromEdges(3, 3, edges)
	at := graphblas.TransposeToCSR[float64](context.Background(), g)

	inf := math.Inf(1)
	d := graphblas.NewDenseVectorN[float64](3)
	d.SetVec(0, 0)
	d.SetVec(1, inf)
	d.SetVec(2, inf)

	next := graphblas.NewDenseVectorN[float64](3)
	graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), at, d, nil, next, binaryop.MinPlus[float64]())

	if !math.IsInf(next.AtVec(0), 1) {
		t.Fatalf("next[0] = %v, want +Inf", next.AtVec(0))
	}
	if next.AtVec(1) != 5 {
		t.Fatalf("next[1] = %v, want 5", next.AtVec(1))
	}
	if next.AtVec(2) != 2 {
		t.Fatalf("next[2] = %v, want 2", next.AtVec(2))
	}
}

// TestSemiringMaxMin checks one widest-path step: capacities into vertex 2
// are min(cap[0], 4) over 0->2 and min(cap[1], 3) over 1->2, reduced with
// max.
func TestSemiringMaxMin(t *testing.T) {
	edges := []graphblas.Edge[float64]{
		{From: 0, To: 2, Weight: 4},
		{From: 1, To: 2, Weight: 3},
	}

	g := graphblas.NewCSRMatrixFromEdges(3, 3, edges)
	at := graphblas.TransposeToCSR[float64](context.Background(), g)

	capacity := graphblas.NewDenseVectorN[float64](3)
	capacity.SetVec(0, 2)
	capacity.SetVec(1, 10)
	capacity.SetVec(2, math.Inf(-1))

	next := graphblas.NewDenseVectorN[float64](3)
	graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), at, capacity, nil, next, binaryop.MaxMin[float64]())

	// Into 2: max(min(4, cap[0]=2), min(3, cap[1]=10)) = max(2, 3) = 3.
	if next.AtVec(2) != 3 {
		t.Fatalf("next[2] = %v, want 3", next.AtVec(2))
	}
}

// TestSemiringDenseMatchesCSR runs the generic (dense-input) path and the
// CSR fast path over the same graph. Both must skip zero entries, so the
// results agree even under min-plus where a participating zero would
// poison the reduction.
func TestSemiringDenseMatchesCSR(t *testing.T) {
	const n = 50
	data, csr, _ := randomSparse(17, n, 6)
	dense := graphblas.NewDenseMatrixFromArrayN(data)

	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, float64(i%9))
	}

	minPlus := binaryop.MinPlus[float64]()

	fromCSR := graphblas.NewDenseVectorN[float64](n)
	graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), csr, x, nil, fromCSR, minPlus)

	fromDense := graphblas.NewDenseVectorN[float64](n)
	graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), dense, x, nil, fromDense, minPlus)

	for i := 0; i < n; i++ {
		if fromCSR.AtVec(i) != fromDense.AtVec(i) {
			t.Fatalf("row %d: csr %v != dense %v", i, fromCSR.AtVec(i), fromDense.AtVec(i))
		}
	}
}

func TestSemiringMasked(t *testing.T) {
	const n = 32
	_, csr, _ := randomSparse(18, n, 5)

	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, 1)
	}

	mask := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i += 2 {
		mask.SetVec(i, 1)
	}

	const sentinel = -777.0
	result := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		result.SetVec(i, sentinel)
	}

	graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), csr, x, mask, result, binaryop.PlusTimes[float64]())

	for i := 0; i < n; i += 2 {
		if result.AtVec(i) != sentinel {
			t.Fatalf("masked row %d was written: %v", i, result.AtVec(i))
		}
	}
}

// TestSemiringCustomInt builds a min-plus semiring for int with an
// explicit identity, since ints have no +Inf.
func TestSemiringCustomInt(t *testing.T) {
	const unreachable = math.MaxInt32

	minPlus := binaryop.NewSemiring[int](
		binaryop.NewMonoID(unreachable, binaryop.Minimum[int]()),
		binaryop.Addition[int](),
	)

	g := graphblas.NewCSRMatrixFromEdges(3, 3, []graphblas.Edge[int]{
		{From: 0, To: 1, Weight: 7},
		{From: 1, To: 2, Weight: 2},
	})
	at := graphblas.TransposeToCSR[int](context.Background(), g)

	d := graphblas.NewDenseVectorN[int](3)
	d.SetVec(0, 0)
	d.SetVec(1, unreachable)
	d.SetVec(2, unreachable)

	next := graphblas.NewDenseVectorN[int](3)
	graphblas.MatrixVectorMultiplyWithSemiring[int](context.Background(), at, d, nil, next, minPlus)

	if next.AtVec(1) != 7 {
		t.Fatalf("next[1] = %v, want 7", next.AtVec(1))
	}
}

func BenchmarkCSRSemiringMinPlus_10000(b *testing.B) {
	const n = 10000
	_, csr, _ := randomSparse(19, n, 10)

	x := graphblas.NewDenseVectorN[float64](n)
	for i := 0; i < n; i++ {
		x.SetVec(i, float64(i%7)+1)
	}
	result := graphblas.NewDenseVectorN[float64](n)
	minPlus := binaryop.MinPlus[float64]()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graphblas.MatrixVectorMultiplyWithSemiring[float64](context.Background(), csr, x, nil, result, minPlus)
	}
}
