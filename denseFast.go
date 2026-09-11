// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas

import (
	"context"

	"github.com/rossmerr/graphblas/constraints"
)

// denseMultiplyFast is a specialized unmasked multiply for dense float64
// matrices. It computes out = s * m row by row (ikj order) over the raw
// backing slices, so each inner step is a contiguous axpy that the SIMD
// kernels can vectorize. Returns false when the operands are not all
// distinct dense float64 matrices of matching shape, leaving the caller
// to run the generic path.
func denseMultiplyFast[T constraints.Number](ctx context.Context, s, m, out Matrix[T]) bool {
	a, ok := any(s).(*DenseMatrixNumber[float64])
	if !ok {
		return false
	}
	b, ok := any(m).(*DenseMatrixNumber[float64])
	if !ok {
		return false
	}
	c, ok := any(out).(*DenseMatrixNumber[float64])
	if !ok {
		return false
	}

	// The kernel overwrites c and reads a and b concurrently row by row.
	if c == a || c == b {
		return false
	}

	if a.c != b.r || c.r != a.r || c.c != b.c {
		return false
	}

	for i := 0; i < a.r; i++ {
		select {
		case <-ctx.Done():
			return true
		default:
		}

		ci := c.data[i]
		for j := range ci {
			ci[j] = 0
		}

		ai := a.data[i]
		for k := 0; k < a.c; k++ {
			v := ai[k]
			if v != 0 {
				axpyFloat64(ci, v, b.data[k][:len(ci)])
			}
		}
	}

	return true
}

// denseAddFast is a specialized unmasked add for dense float64 matrices,
// computing out = s + m one contiguous row at a time. Returns false when
// the operands are not all dense float64 matrices of matching shape.
func denseAddFast[T constraints.Number](ctx context.Context, s, m, out Matrix[T]) bool {
	a, ok := any(s).(*DenseMatrixNumber[float64])
	if !ok {
		return false
	}
	b, ok := any(m).(*DenseMatrixNumber[float64])
	if !ok {
		return false
	}
	c, ok := any(out).(*DenseMatrixNumber[float64])
	if !ok {
		return false
	}

	if a.r != b.r || a.c != b.c || c.r != a.r || c.c != a.c {
		return false
	}

	for i := 0; i < a.r; i++ {
		select {
		case <-ctx.Done():
			return true
		default:
		}

		ci := c.data[i]
		addFloat64(ci, a.data[i][:len(ci)], b.data[i][:len(ci)])
	}

	return true
}
