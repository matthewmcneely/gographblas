# GraphBLAS

![Go](https://github.com/rossmerr/graphblas/workflows/Go/badge.svg)
[![Go Report Card](https://goreportcard.com/badge/github.com/rossmerr/graphblas)](https://goreportcard.com/report/github.com/rossmerr/graphblas)
[![Read the Docs](https://pkg.go.dev/badge/golang.org/x/pkgsite)](https://pkg.go.dev/github.com/rossmerr/graphblas)

A sparse linear algebra library implementing may of the ideas from the [GraphBLAS Forum](https://graphblas.github.io/) in Go.

## About this fork

This fork moves the module to Go 1.27 and adds specialized kernels for unmasked dense `float64` operations, with optional SIMD variants built on Go 1.27's experimental portable [`simd` package](https://go.dev/doc/go1.27).

What changed:

- `Multiply` and `Add` on dense `float64` matrices with a nil mask now run flat row kernels (`denseFast.go`) instead of the per-element interface path. Multiplication uses ikj ordering, so the inner step is a contiguous axpy over each output row.
- The axpy/add kernels have two build variants: `kernels_simd.go` under `//go:build goexperiment.simd`, and scalar fallbacks in `kernels_scalar.go`. The default build has no dependency on the experimental API.
- Everything else (masked operations, sparse formats, other element types) is unchanged and falls through to the original generic path.

Measured on an Apple M4 Pro with `go1.27.0`, 100x100 dense `float64`:

| Benchmark | before (interface path) | scalar kernels (default build) | `GOEXPERIMENT=simd` |
|---|---|---|---|
| dense multiply | 8-11 ms, 20,303 allocs | **287 µs, 102 allocs** | 406 µs |
| dense add | 102 µs | **18 µs** | 21 µs |
| `axpyFloat64`, n=4096 | n/a | **1.07 µs** | 1.51 µs |
| `axpyFloat32`, n=4096 | n/a | 1.07 µs | **0.78 µs** |

The headline is that the kernel restructure delivers the ~30x, not the vector instructions. On 128-bit NEON, `float64` SIMD is slower than the scalar kernels: each vector op carries only 2 lanes but pays slice and bounds-check overhead per load/store plus a runtime width-dispatch call, while Go's arm64 backend already fuses the scalar loop into `FMADD`. `float32` (4 lanes) is where SIMD starts to win, and the wider amd64 vectors (AVX2, AVX-512) should shift the balance further, though that build is unmeasured here. Sparse (CSR/CSC) kernels stay scalar on every architecture: the Go 1.27 `simd` packages ship no gather/scatter.

Run the comparison yourself (requires Go 1.27):

```sh
go test -bench 'Axpy|Dense' .                     # scalar kernels
GOEXPERIMENT=simd go test -bench 'Axpy|Dense' .   # SIMD kernels
GOEXPERIMENT=simd go test -run TestSIMDInfo -v .  # logs vector width + emulation status
```

The `simd` package is experimental and its API may change, which is why the SIMD variants are opt-in and the default build stays scalar.

Sparse Matrix Formats:
Compressed Sparse Row (CSR)
Compressed Sparse Column (CSC)
Sparse Vector

Supports bool | int | int8 | int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | uintptr | float32 | float64

```go
array := [][]float64{
		[]float64{0, 0, 0, 1, 0, 0, 0},
		[]float64{1, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 0, 1, 0, 1, 1},
		[]float64{1, 0, 0, 0, 0, 0, 1},
		[]float64{0, 1, 0, 0, 0, 0, 1},
		[]float64{0, 0, 1, 0, 1, 0, 0},
		[]float64{0, 1, 0, 0, 0, 0, 0},
    }

g := graphblas.NewDenseMatrixFromArrayN(array)

atx := breadthfirst.Search[float64](context.Background(), g, 3, func(i graphblas.Vector[float64]) bool {
    return i.AtVec(5) == 1
})
```
