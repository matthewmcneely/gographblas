// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

func TestSIMDInfo(t *testing.T) {
	t.Log(simdInfo())
}

// The kernels and the math.FMA references legitimately differ: scalar
// amd64 builds round the multiply and add separately (the compiler only
// contracts to FMA on arm64), while the SIMD kernels and arm64 scalar
// code fuse. That difference is a few ulps of the operands, but when the
// sum cancels to near zero it is unbounded relative to the result, so
// both comparisons carry an absolute term sized to the O(1) test data in
// addition to the relative one.

func almostEqual(a, b float64) bool {
	if a == b {
		return true
	}
	diff := math.Abs(a - b)
	scale := math.Max(math.Abs(a), math.Abs(b))
	return diff <= 1e-12*scale+1e-12
}

func almostEqual32(a, b float32) bool {
	if a == b {
		return true
	}
	diff := math.Abs(float64(a) - float64(b))
	scale := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	return diff <= 1e-6*scale+1e-6
}

// Lengths chosen to cover empty slices, tails shorter than a vector, and
// exact multiples of every plausible lane count (2, 4, 8).
var kernelLengths = []int{0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 63, 64, 100, 129}

func TestAxpyFloat64(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	for _, n := range kernelLengths {
		dst := make([]float64, n)
		x := make([]float64, n)
		want := make([]float64, n)
		for i := 0; i < n; i++ {
			dst[i] = rnd.NormFloat64()
			x[i] = rnd.NormFloat64()
		}
		alpha := rnd.NormFloat64()
		for i := 0; i < n; i++ {
			want[i] = math.FMA(x[i], alpha, dst[i])
		}

		axpyFloat64(dst, alpha, x)

		for i := 0; i < n; i++ {
			if !almostEqual(dst[i], want[i]) {
				t.Fatalf("n=%d: dst[%d] = %v, want %v", n, i, dst[i], want[i])
			}
		}
	}
}

func TestAxpyFloat32(t *testing.T) {
	rnd := rand.New(rand.NewSource(2))
	for _, n := range kernelLengths {
		dst := make([]float32, n)
		x := make([]float32, n)
		want := make([]float32, n)
		for i := 0; i < n; i++ {
			dst[i] = float32(rnd.NormFloat64())
			x[i] = float32(rnd.NormFloat64())
		}
		alpha := float32(rnd.NormFloat64())
		for i := 0; i < n; i++ {
			want[i] = float32(math.FMA(float64(x[i]), float64(alpha), float64(dst[i])))
		}

		axpyFloat32(dst, alpha, x)

		for i := 0; i < n; i++ {
			if !almostEqual32(dst[i], want[i]) {
				t.Fatalf("n=%d: dst[%d] = %v, want %v", n, i, dst[i], want[i])
			}
		}
	}
}

func TestAddFloat64(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	for _, n := range kernelLengths {
		dst := make([]float64, n)
		x := make([]float64, n)
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			x[i] = rnd.NormFloat64()
			y[i] = rnd.NormFloat64()
		}

		addFloat64(dst, x, y)

		for i := 0; i < n; i++ {
			if dst[i] != x[i]+y[i] {
				t.Fatalf("n=%d: dst[%d] = %v, want %v", n, i, dst[i], x[i]+y[i])
			}
		}

		// Aliased: dst is also the first operand.
		copy(dst, x)
		addFloat64(dst, dst, y)
		for i := 0; i < n; i++ {
			if dst[i] != x[i]+y[i] {
				t.Fatalf("n=%d aliased: dst[%d] = %v, want %v", n, i, dst[i], x[i]+y[i])
			}
		}
	}
}

func randomDense(rnd *rand.Rand, r, c int) *DenseMatrixNumber[float64] {
	m := NewDenseMatrixN[float64](r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, rnd.NormFloat64())
		}
	}
	return m
}

// TestDenseMultiplyFastMatchesReference checks the specialized dense
// multiply (SIMD or scalar depending on build) against a naive triple
// loop, using shapes that exercise partial-vector tails.
func TestDenseMultiplyFastMatchesReference(t *testing.T) {
	rnd := rand.New(rand.NewSource(4))
	shapes := [][3]int{{1, 1, 1}, {3, 5, 7}, {37, 23, 41}, {64, 64, 64}, {100, 100, 100}}

	for _, shape := range shapes {
		r, k, c := shape[0], shape[1], shape[2]
		a := randomDense(rnd, r, k)
		b := randomDense(rnd, k, c)

		got := NewDenseMatrixN[float64](r, c)
		if !denseMultiplyFast[float64](context.Background(), a, b, got) {
			t.Fatalf("shape %v: fast path unexpectedly declined", shape)
		}

		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				want := 0.0
				for l := 0; l < k; l++ {
					want += a.At(i, l) * b.At(l, j)
				}
				if !almostEqual(got.At(i, j), want) {
					t.Fatalf("shape %v: C[%d][%d] = %v, want %v", shape, i, j, got.At(i, j), want)
				}
			}
		}
	}
}

// TestDenseMultiplyPublicAPI makes sure the fast path wired into multiply()
// produces the same result the generic masked path produces.
func TestDenseMultiplyPublicAPI(t *testing.T) {
	rnd := rand.New(rand.NewSource(5))
	a := randomDense(rnd, 37, 23)
	b := randomDense(rnd, 23, 41)

	fast := a.Multiply(b)

	slow := NewDenseMatrixN[float64](37, 41)
	MatrixMatrixMultiply[float64](context.Background(), a, b, NewEmptyMask(37, 41), slow)

	for i := 0; i < 37; i++ {
		for j := 0; j < 41; j++ {
			if !almostEqual(fast.At(i, j), slow.At(i, j)) {
				t.Fatalf("C[%d][%d]: fast %v != slow %v", i, j, fast.At(i, j), slow.At(i, j))
			}
		}
	}
}

func TestDenseAddPublicAPI(t *testing.T) {
	rnd := rand.New(rand.NewSource(6))
	a := randomDense(rnd, 37, 41)
	b := randomDense(rnd, 37, 41)

	fast := a.Add(b)

	for i := 0; i < 37; i++ {
		for j := 0; j < 41; j++ {
			want := a.At(i, j) + b.At(i, j)
			if fast.At(i, j) != want {
				t.Fatalf("C[%d][%d] = %v, want %v", i, j, fast.At(i, j), want)
			}
		}
	}
}

func benchmarkAxpyFloat64(b *testing.B, n int) {
	dst := make([]float64, n)
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
	}
	b.SetBytes(int64(16 * n))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axpyFloat64(dst, 1.5, x)
	}
}

func BenchmarkAxpyFloat64_128(b *testing.B)  { benchmarkAxpyFloat64(b, 128) }
func BenchmarkAxpyFloat64_4096(b *testing.B) { benchmarkAxpyFloat64(b, 4096) }

func BenchmarkAxpyFloat32_4096(b *testing.B) {
	n := 4096
	dst := make([]float32, n)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i)
	}
	b.SetBytes(int64(8 * n))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axpyFloat32(dst, 1.5, x)
	}
}

func benchmarkDenseMultiply(b *testing.B, n int) {
	rnd := rand.New(rand.NewSource(7))
	x := randomDense(rnd, n, n)
	y := randomDense(rnd, n, n)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Multiply(y)
	}
}

func BenchmarkDenseMultiply_100(b *testing.B) { benchmarkDenseMultiply(b, 100) }
func BenchmarkDenseMultiply_400(b *testing.B) { benchmarkDenseMultiply(b, 400) }

func BenchmarkDenseAdd_100(b *testing.B) {
	rnd := rand.New(rand.NewSource(8))
	x := randomDense(rnd, 100, 100)
	y := randomDense(rnd, 100, 100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Add(y)
	}
}
