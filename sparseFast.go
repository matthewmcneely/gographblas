// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas

import (
	"context"

	"github.com/rossmerr/graphblas/constraints"
)

// sparseMatrixVectorFast is a specialized matrix-vector multiply for CSR
// and CSC matrices against a dense vector. It walks the compressed arrays
// directly, so the work is proportional to the stored entries instead of
// the generic path's full row scan with a vector allocation per output
// row. The mask is honored per output row. Returns false when the operand
// types or shapes do not match, leaving the caller to run the generic
// path.
func sparseMatrixVectorFast[T constraints.Number](ctx context.Context, s, m Matrix[T], mask Mask, out Matrix[T]) bool {
	x, ok := any(m).(*DenseVectorNumber[T])
	if !ok {
		return false
	}

	y, ok := any(out).(*DenseVectorNumber[T])
	if !ok {
		return false
	}

	// The kernels read x while writing y.
	if x == y {
		return false
	}

	switch a := any(s).(type) {
	case *CSRMatrix[T]:
		return csrMatrixVector(ctx, a, x, mask, y)
	case *CSCMatrix[T]:
		return cscMatrixVector(ctx, a, x, mask, y)
	}

	return false
}

// csrMatrixVector computes y = a * x row by row: each output element is a
// gather over the stored entries of its row.
func csrMatrixVector[T constraints.Number](ctx context.Context, a *CSRMatrix[T], x *DenseVectorNumber[T], mask Mask, y *DenseVectorNumber[T]) bool {
	if a.c != x.l || y.l != a.r {
		return false
	}

	xs := x.values
	_, unmasked := mask.(*EmptyMask)

	for r := 0; r < a.r; r++ {
		if r&1023 == 0 {
			select {
			case <-ctx.Done():
				return true
			default:
			}
		}

		sum := Default[T]()
		for j := a.rowStart[r]; j < a.rowStart[r+1]; j++ {
			sum += a.values[j] * xs[a.cols[j]]
		}

		if unmasked || !mask.Element(r, 0) {
			y.values[r] = sum
		}
	}

	return true
}

// cscMatrixVector computes y = a * x column by column, scattering each
// x[c] across the stored entries of column c into a scratch accumulator,
// then applying the mask on the final copy.
func cscMatrixVector[T constraints.Number](ctx context.Context, a *CSCMatrix[T], x *DenseVectorNumber[T], mask Mask, y *DenseVectorNumber[T]) bool {
	if a.c != x.l || y.l != a.r {
		return false
	}

	sums := make([]T, a.r)
	for c := 0; c < a.c; c++ {
		if c&1023 == 0 {
			select {
			case <-ctx.Done():
				return true
			default:
			}
		}

		v := x.values[c]
		if v == Default[T]() {
			continue
		}

		for j := a.colStart[c]; j < a.colStart[c+1]; j++ {
			sums[a.rows[j]] += a.values[j] * v
		}
	}

	if _, unmasked := mask.(*EmptyMask); unmasked {
		copy(y.values, sums)
		return true
	}

	for r := 0; r < a.r; r++ {
		if !mask.Element(r, 0) {
			y.values[r] = sums[r]
		}
	}

	return true
}
