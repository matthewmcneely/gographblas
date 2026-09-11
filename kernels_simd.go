// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//go:build goexperiment.simd

package graphblas

import (
	"fmt"
	"simd"
)

// simdInfo reports which kernel implementation this build uses.
func simdInfo() string {
	return fmt.Sprintf("goexperiment.simd on: vector=%d bits, emulated=%v", simd.VectorBitSize(), simd.Emulated())
}

// axpyFloat64 computes dst[i] += alpha * x[i] for i in range of dst.
// x must be at least as long as dst.
//
// The main loop is unrolled 4x: with 128-bit vectors each Load/Store
// carries a few instructions of slice and bounds overhead, which at two
// float64 lanes per vector would otherwise cancel out the SIMD gain.
func axpyFloat64(dst []float64, alpha float64, x []float64) {
	a := simd.BroadcastFloat64s(alpha)
	lanes := a.Len()
	i := 0
	for ; i+4*lanes <= len(dst); i += 4 * lanes {
		x0 := simd.LoadFloat64s(x[i:])
		x1 := simd.LoadFloat64s(x[i+lanes:])
		x2 := simd.LoadFloat64s(x[i+2*lanes:])
		x3 := simd.LoadFloat64s(x[i+3*lanes:])
		d0 := simd.LoadFloat64s(dst[i:])
		d1 := simd.LoadFloat64s(dst[i+lanes:])
		d2 := simd.LoadFloat64s(dst[i+2*lanes:])
		d3 := simd.LoadFloat64s(dst[i+3*lanes:])
		x0.MulAdd(a, d0).Store(dst[i:])
		x1.MulAdd(a, d1).Store(dst[i+lanes:])
		x2.MulAdd(a, d2).Store(dst[i+2*lanes:])
		x3.MulAdd(a, d3).Store(dst[i+3*lanes:])
	}
	for ; i+lanes <= len(dst); i += lanes {
		xv := simd.LoadFloat64s(x[i:])
		dv := simd.LoadFloat64s(dst[i:])
		xv.MulAdd(a, dv).Store(dst[i:])
	}
	if i < len(dst) {
		xv, _ := simd.LoadFloat64sPart(x[i:len(dst)])
		dv, _ := simd.LoadFloat64sPart(dst[i:])
		xv.MulAdd(a, dv).StorePart(dst[i:])
	}
}

// axpyFloat32 computes dst[i] += alpha * x[i] for i in range of dst.
// x must be at least as long as dst.
func axpyFloat32(dst []float32, alpha float32, x []float32) {
	a := simd.BroadcastFloat32s(alpha)
	lanes := a.Len()
	i := 0
	for ; i+lanes <= len(dst); i += lanes {
		xv := simd.LoadFloat32s(x[i:])
		dv := simd.LoadFloat32s(dst[i:])
		xv.MulAdd(a, dv).Store(dst[i:])
	}
	if i < len(dst) {
		xv, _ := simd.LoadFloat32sPart(x[i:len(dst)])
		dv, _ := simd.LoadFloat32sPart(dst[i:])
		xv.MulAdd(a, dv).StorePart(dst[i:])
	}
}

// addFloat64 computes dst[i] = x[i] + y[i] for i in range of dst.
// x and y must be at least as long as dst. dst may alias x or y.
func addFloat64(dst, x, y []float64) {
	lanes := simd.VectorBitSize() / 64
	i := 0
	for ; i+lanes <= len(dst); i += lanes {
		simd.LoadFloat64s(x[i:]).Add(simd.LoadFloat64s(y[i:])).Store(dst[i:])
	}
	if i < len(dst) {
		xv, _ := simd.LoadFloat64sPart(x[i:len(dst)])
		yv, _ := simd.LoadFloat64sPart(y[i:len(dst)])
		xv.Add(yv).StorePart(dst[i:])
	}
}
