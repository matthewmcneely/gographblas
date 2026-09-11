// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//go:build !goexperiment.simd

package graphblas

// simdInfo reports which kernel implementation this build uses.
func simdInfo() string {
	return "goexperiment.simd off: scalar kernels"
}

// axpyFloat64 computes dst[i] += alpha * x[i] for i in range of dst.
// x must be at least as long as dst.
func axpyFloat64(dst []float64, alpha float64, x []float64) {
	x = x[:len(dst)]
	for i := range dst {
		dst[i] += alpha * x[i]
	}
}

// axpyFloat32 computes dst[i] += alpha * x[i] for i in range of dst.
// x must be at least as long as dst.
func axpyFloat32(dst []float32, alpha float32, x []float32) {
	x = x[:len(dst)]
	for i := range dst {
		dst[i] += alpha * x[i]
	}
}

// addFloat64 computes dst[i] = x[i] + y[i] for i in range of dst.
// x and y must be at least as long as dst. dst may alias x or y.
func addFloat64(dst, x, y []float64) {
	x = x[:len(dst)]
	y = y[:len(dst)]
	for i := range dst {
		dst[i] = x[i] + y[i]
	}
}
