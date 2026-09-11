# GraphBLAS

![Go](https://github.com/rossmerr/graphblas/workflows/Go/badge.svg)
[![Go Report Card](https://goreportcard.com/badge/github.com/rossmerr/graphblas)](https://goreportcard.com/report/github.com/rossmerr/graphblas)
[![Read the Docs](https://pkg.go.dev/badge/golang.org/x/pkgsite)](https://pkg.go.dev/github.com/rossmerr/graphblas)

A sparse linear algebra library implementing may of the ideas from the [GraphBLAS Forum](https://graphblas.github.io/) in Go.

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

## Algorithms

The examples below reuse `g` from the snippet above and assume `ctx := context.Background()`.

### Breadth-first search (`breadthfirst`)

`Search` runs a level-synchronous BFS from a source vertex, expressed the GraphBLAS way: the frontier is a vector, each level is one masked matrix-vector multiply, and the visited vector masks off vertices already seen. The callback receives the new frontier after every level and returns true to stop the traversal; the return value is the frontier at the level where the search stopped.

```go
// Traverse from vertex 3, stopping once vertex 5 is reachable.
frontier := breadthfirst.Search[float64](ctx, g, 3, func(v graphblas.Vector[float64]) bool {
	return v.AtVec(5) > 0
})
```

### PageRank (`centrality`)

`PageRank` ranks vertices by the stationary probability that a random surfer occupies them, computed by power iteration. Each vertex spreads its rank across its out-links in proportion to edge weight, rank held by dangling vertices is redistributed uniformly, and iteration stops when the L1 change drops below the tolerance. The returned ranks sum to 1.

```go
ranks := centrality.PageRank[float64](ctx, g) // damping 0.85, tolerance 1e-6, max 100 iterations
ranks = centrality.PageRankWithOptions[float64](ctx, g, 0.9, 1e-9, 200)
```

### Single-source shortest path (`shortestPath`)

`SingleSource` computes the distance from a source vertex to every other vertex by Bellman-Ford edge relaxation. `a.At(u, v)` is the weight of edge `u -> v` and zero marks an absent edge; unreachable vertices report `+Inf`. Negative weights are supported on graphs without negative cycles. Note the package name is `shortestpath` while the import path ends in `shortestPath`.

```go
import shortestpath "github.com/rossmerr/graphblas/shortestPath"

dist := shortestpath.SingleSource[float64](ctx, g, 0)
unreachable := math.IsInf(dist.AtVec(4), 1)
```

### GraphBLAS primitives (root package)

The building blocks the algorithms compose, most of which accept an optional mask to control which output cells are written:

- Multiplication: `MatrixMatrixMultiply` (mxm), `MatrixVectorMultiply` (mxv), `VectorMatrixMultiply` (vxm)
- Element-wise: `ElementWiseMatrixMultiply`, `ElementWiseMatrixAdd`, and their vector forms, plus `Add`, `Subtract`, `Scalar`, `Negative`
- Structural: `Transpose`, `TransposeToCSR`, `TransposeToCSC`, `Equal`, `NotEqual`
- Reductions: `ReduceMatrixToVector`, `ReduceMatrixToScalar`, and `WithMonoID` variants that take a custom monoid from the `binaryop` package
- `Apply` maps a `unaryop.UnaryOp` over every element

```go
result := graphblas.NewDenseVectorN[float64](g.Rows())
graphblas.MatrixVectorMultiply[float64](ctx, g, frontier, nil, result)
```

### Strassen multiplication (`math/strassen`)

Divide-and-conquer matrix multiplication that trades 8 recursive block multiplies for 7 plus extra additions. At or below the crossover size (default 64) it switches to standard `MatrixMatrixMultiply`, which on this fork lands in the dense `float64` fast path.

```go
c := strassen.Multiply[float64](ctx, a, b)
c = strassen.MultiplyCrossoverPoint[float64](ctx, a, b, 128) // tune the switch-over size
```

### Reduced row echelon form (`math/reduced`)

Gauss-Jordan elimination. Returns a new matrix and leaves the input untouched.

```go
r := reduced.Reduced[float64](m)
```

### Matrix predicates (`math/symmetric`, `math/skewsymmetric`)

Square-matrix checks built on `Transpose` and `Equal`.

```go
symmetric.Symmetric[float64](m)         // true when m equals its transpose
skewsymmetric.SkewSymmetric[float64](m) // true when m equals the negative of its transpose
```

### Sorting (`sort`)

`BubbleRow` and `BubbleColumns` sort the rows or columns of a rune matrix lexicographically, comparing whole rows or columns with `graphblas.Compare`.

```go
sorted := sort.BubbleRow(ctx, words) // words is a graphblas.MatrixRune
```

### Not yet implemented

Betweenness and closeness centrality, the `clustering` package (Markov, spectral, peer pressure, local), and the all-pairs and temporal shortest-path variants are declaration-only placeholders inherited from upstream: the files compile but contain no implementations. The primitives above are the pieces those algorithms would compose from.

## About this fork

This fork moves the module to Go 1.27 and adds specialized kernels for unmasked dense `float64` operations, with optional SIMD variants built on Go 1.27's experimental portable [`simd` package](https://go.dev/doc/go1.27).

What changed:

- `Multiply` and `Add` on dense `float64` matrices with a nil mask now run flat row kernels (`denseFast.go`) instead of the per-element interface path. Multiplication uses ikj ordering, so the inner step is a contiguous axpy over each output row.
- The axpy/add kernels have two build variants: `kernels_simd.go` under `//go:build goexperiment.simd`, and scalar fallbacks in `kernels_scalar.go`. The default build has no dependency on the experimental API.
- Everything else (masked operations, sparse formats, other element types) is unchanged and falls through to the original generic path.
- This fork also implements PageRank and single-source shortest path (see Algorithms above) and fixes an upstream `SparseVector` index-lookup bug that compared an element index against the stored entry count, which made high-index entries unreadable and silently dropped terms from sparse multiplies.
- CSR and CSC matrix-vector multiplies take a dedicated path (`sparseFast.go`) that walks the compressed arrays directly, works for every element type, and honors masks per output row. At 10,000 vertices and ~100k edges, one mxv drops from 1.04 s and 825 MB allocated (the generic path materializes and binary-searches a vector per output row) to 73 µs and 16 B for CSR, roughly 14,000x, with CSC at 106 µs. This is what makes PageRank and breadth-first search practical on large graphs.

Measured on an Apple M4 Pro with `go1.27.0`, 100x100 dense `float64`:

| Benchmark | before (interface path) | scalar kernels (default build) | `GOEXPERIMENT=simd` |
|---|---|---|---|
| dense multiply | 8-11 ms, 20,303 allocs | **287 µs, 102 allocs** | 406 µs |
| dense add | 102 µs | **18 µs** | 21 µs |
| `axpyFloat64`, n=4096 | n/a | **1.07 µs** | 1.51 µs |
| `axpyFloat32`, n=4096 | n/a | 1.07 µs | **0.78 µs** |

The headline on arm64 is that the kernel restructure delivers the ~30x, not the vector instructions. On 128-bit NEON, `float64` SIMD is slower than the scalar kernels: each vector op carries only 2 lanes but pays slice and bounds-check overhead per load/store plus a runtime width-dispatch call, while Go's arm64 backend already fuses the scalar loop into `FMADD`. `float32` (4 lanes) is where SIMD starts to win on that hardware. Wider amd64 vectors flip the `float64` verdict outright.

The same branch on a GCP `c4d-standard-8` (AMD EPYC 9B45, Zen 5, full-width AVX-512), where Go selects `vector=512 bits`:

| Benchmark | scalar kernels (default build) | `GOEXPERIMENT=simd` | speedup |
|---|---|---|---|
| dense multiply, 100x100 | 267 µs | **188 µs** | 1.4x |
| dense multiply, 400x400 | 17.1 ms | **10.3 ms** | 1.7x |
| dense add, 100x100 | 20 µs | 20 µs | 1.0x |
| `axpyFloat64`, n=4096 | 1.55 µs | **0.60 µs** | 2.6x |
| `axpyFloat32`, n=4096 | 1.50 µs | **0.28 µs** | 5.3x |

At 8 `float64` lanes the per-op overhead amortizes and the kernels pull ahead. The `float32` gain is larger than the lane ratio suggests because the scalar baseline is op-bound: scalar `float32` axpy runs no faster than scalar `float64`, so SIMD is what converts the narrower element into throughput. End-to-end multiply improves less than the kernel (1.4x to 1.7x) because the remaining time sits outside the axpy inner loop: output-row zeroing, the runtime width-dispatch call per kernel invocation, and row-pointer chasing in the `[][]float64` layout. Dense add is unchanged since it is dominated by the output copy and its allocations rather than arithmetic. A `GOAMD64=v3` control run leaves the scalar kernels unchanged at n=4096 and improves the n=128 case by about 13%: Go's amd64 backend emits separate multiply and add instructions even at v3 rather than contracting to FMA, so the kernel-level gap is genuine vector advantage rather than an artifact of the default `v1` baseline. The SIMD build is unaffected by `GOAMD64` since the kernels detect vector width at runtime.

Sparse (CSR/CSC) kernels stay scalar on every architecture: the Go 1.27 `simd` packages ship no gather/scatter.

Run the comparison yourself (requires Go 1.27):

```sh
go test -bench 'Axpy|Dense' .                     # scalar kernels
GOEXPERIMENT=simd go test -bench 'Axpy|Dense' .   # SIMD kernels
GOEXPERIMENT=simd go test -run TestSIMDInfo -v .  # logs vector width + emulation status
```

The `simd` package is experimental and its API may change, which is why the SIMD variants are opt-in and the default build stays scalar.

