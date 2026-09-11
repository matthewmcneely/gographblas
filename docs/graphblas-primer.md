# A GraphBLAS primer

This library implements ideas from the [GraphBLAS Forum](https://graphblas.github.io/): graph algorithms expressed as sparse linear algebra. If you know graphs but the matrix framing is new, this primer walks the core ideas and then maps them onto this package, noting along the way where the code genuinely computes with matrix algebra and where it deliberately does not.

## The big idea

A directed graph of n vertices is an n x n matrix `A` where `A[u][v]` holds the weight of the edge `u -> v`, and the absence of an edge is the absence of an entry. A set of vertices with per-vertex values is a vector: a BFS frontier, a distance estimate, a PageRank score.

The payoff is that one matrix-vector multiply is one synchronized step of a graph traversal. Unpack the definition:

```
y[v] = SUM over u of ( A'[v][u] * x[u] )        where A' is A transposed
```

Read as a graph: every vertex `v` looks at its incoming edges, combines each edge weight with the value sitting at the edge's source, and reduces the results into one value. That is "gather from your neighbors" for every vertex at once. An algorithm that repeats a neighborhood step until it stabilizes, which describes most of classical graph analytics, becomes a loop around one multiply.

The second idea is the one that gives GraphBLAS its power: nothing about that definition requires SUM and `*` to be arithmetic. Replace them with any pair (⊕, ⊗) forming a semiring and the same multiply computes a different algorithm:

| Semiring | ⊕ (reduce) | ⊗ (combine) | `y = A'x` means | Algorithm |
|---|---|---|---|---|
| plus-times | `+` | `*` | weighted sum of neighbor values | PageRank's pull step |
| min-plus (tropical) | `min` | `+` | cheapest way to reach me | Bellman-Ford relaxation |
| or-and (boolean) | `or` | `and` | any neighbor is reachable | BFS reachability |
| max-min | `max` | `min` | widest bottleneck into me | widest path |

The third idea is masks: a multiply (or other operation) can take a mask that controls which output cells are written, leaving the rest untouched. BFS uses this to keep already-visited vertices out of the next frontier without maintaining a visited set in algorithm code.

Why organize algorithms this way at all? Separation of concerns. The algebra is a small, fixed vocabulary, so all the performance work (sparse storage formats, cache-friendly kernels, vectorization, parallelism) happens once, inside the multiply, and every algorithm built on it benefits. The algorithms themselves shrink to a few lines that read like math.

## How this library represents graphs

Everything is a `Matrix[T]` (generic over the numeric types), with vectors as n x 1 matrices implementing `Vector[T]`. Four storage shapes matter:

- `DenseMatrix` / `DenseVector`: a full 2D array / slice. Right when most cells are meaningful.
- `CSRMatrix` (compressed sparse row): three flat arrays. `rowStart[r]` points into `cols`/`values`, which hold each row's stored entries sorted by column. Walking a vertex's out-edges is a contiguous slice scan, which makes CSR the natural adjacency format.
- `CSCMatrix` (compressed sparse column): the same idea organized by column, so in-edges are contiguous.
- `SparseVector`: parallel `indices`/`values` arrays.

A concrete CSR example, the graph `0 -> 1 (w 2)`, `0 -> 2 (w 3)`, `2 -> 1 (w 4)`:

```
rowStart: [0, 2, 2, 3]      row r's entries live at positions rowStart[r] .. rowStart[r+1]
cols:     [1, 2, 1]
values:   [2, 3, 4]
```

One convention runs through the whole library: **a zero value marks an absent edge**. Sparse formats refuse to store zeros, and the algorithms skip them, so an explicit zero-weight edge is not representable. Keep that in mind when weights can legitimately be zero (shift them, or use a semiring whose "nothing" differs from 0).

Build graphs from edge lists rather than 2D arrays:

```go
g := graphblas.NewCSRMatrixFromEdges(n, n, []graphblas.Edge[float64]{
	{From: 0, To: 1, Weight: 2},
	{From: 0, To: 2, Weight: 3},
	{From: 2, To: 1, Weight: 4},
})
```

`NewCSRMatrixFromEdges` bulk-builds in one pass (duplicates sum their weights). Per-element `Set` on compressed formats splices arrays and is only sensible for small or incremental updates.

## The operations, and whether they are matrix algebra

### The multiply family

`MatrixMatrixMultiply` (mxm), `MatrixVectorMultiply` (mxv), and `VectorMatrixMultiply` (vxm) compute the plus-times product, with an optional mask. Two things are worth knowing:

- The semiring is **hardcoded to plus-times** in these functions (`sum += a * x` in the kernel). They are matrix algebra, but only the arithmetic kind.
- Both vector variants compute `A . x`. The direction convention this implies: the result at vertex `v` gathers along the *rows* of the matrix, so with the `A[from][to]` edge convention a frontier advances along **incoming** edges. Algorithms that want to walk edge direction transpose first (see BFS below). This is the most commonly misunderstood thing in the package.

Mathematically these are matrix products; mechanically, the implementation type-switches to fast paths that walk raw arrays (`denseFast.go`, `sparseFast.go`) and falls back to a generic interface-driven path for anything else. That split is normal for GraphBLAS libraries: the algebra is the contract, the kernels are plumbing. Hardware-wise, the dense `float64` kernels carry optional SIMD variants while the sparse kernels are scalar on every architecture; the execution-tier table at the end of this primer says which is which and why.

### The semiring multiply

`MatrixVectorMultiplyWithSemiring` is the real GraphBLAS move: the same mxv parameterized by a `binaryop.Semiring[T]`, which pairs an additive monoid (operator plus its identity, the seed of each reduction) with a multiplicative operator. Stock semirings are `binaryop.PlusTimes`, `binaryop.MinPlus`, and `binaryop.MaxMin`; `binaryop.NewSemiring` builds custom ones, such as min-plus over `int` with an explicit "infinity".

```go
// One Bellman-Ford relaxation step: from d = [0, +Inf, +Inf],
// one (min, +) multiply by the transpose yields one-hop distances.
graphblas.MatrixVectorMultiplyWithSemiring[float64](ctx, at, dist, nil, next, binaryop.MinPlus[float64]())
```

Semantics to remember: only stored, non-zero matrix entries participate (the zero-means-absent convention, applied uniformly to dense and sparse inputs), while vector entries always participate, since a zero vector element is not an identity for an arbitrary semiring. The distance 0 at a shortest-path source is precisely the value that must flow.

### Element-wise operations

`ElementWiseMatrixAdd`/`ElementWiseMatrixMultiply` (and vector forms) combine two operands cell by cell, GraphBLAS's eWiseAdd/eWiseMult. `Add`, `Subtract`, `Scalar`, and `Negative` are the arithmetic versions. These are algebra in the trivial sense: no reduction structure, just per-cell combination.

### Masks

Any `Mask` (anything with `Element(r, c) bool`) can gate an operation's output; `Element` returning true means the cell is **not** written. Any numeric matrix or vector works as a mask, reading positive values as true, which is how BFS passes its visited vector directly. A nil mask writes everything.

### Transpose

`Transpose` writes the flipped matrix through the generic interface; `TransposeToCSR` collects entries and bulk-builds, which is the one to reach for on anything large. Transposition matters more here than in ordinary linear algebra because of the direction convention above: it converts "gather along in-edges" into "advance along out-edges".

### Reductions and Apply

`ReduceMatrixToVector`/`ReduceMatrixToScalar` fold a matrix with a `binaryop.MonoID` (default: max for vectors, plus for scalars), and `Apply` maps a `unaryop.UnaryOp` over the elements. One implementation honesty note: the monoid reductions stream elements through a channel to a folding goroutine, one send per element. Fine for occasional summaries, but nothing hot-path-shaped, which is why the algorithms below reduce with plain loops.

## The algorithms through this lens

Where each shipped algorithm actually sits on the "is it matrix algebra?" spectrum:

| Algorithm | Matrix algebra? | The compute step |
|---|---|---|
| Breadth-first search | yes | one masked plus-times mxv per level |
| Single-source shortest path | yes | one min-plus semiring mxv per round |
| PageRank | mostly | one mxv per iteration; scalar vector loops around it |
| `Between` (Dijkstra) | deliberately no | binary heap and adjacency scan |
| Strassen multiplication | yes, by definition | recursive block mxm |
| Reduced row echelon form | classic dense linear algebra | row operations via the matrix interface |

**Breadth-first search** (`breadthfirst.Search`) is the textbook GraphBLAS loop: the frontier is a vector, each level is `MatrixVectorMultiply(a, frontier, visited, result)` with the visited vector as the mask, then visited absorbs the result. Remember the direction convention: `Search` advances along the matrix's columns, so to traverse a conventionally-built `A[from][to]` graph in edge direction, hand it the transpose (the package example does exactly this).

**Single-source shortest path** (`shortestpath.SingleSource`) is Bellman-Ford in its GraphBLAS form and the best illustration in the repo. Transpose once, then each round is

```
next = at (min,+) dist ;  dist = min(dist, next) elementwise
```

stopping when a round changes nothing or after n-1 rounds. The k-th round holds shortest distances using at most k edges. Nothing about the relaxation logic lives in algorithm code; the semiring is the algorithm.

**PageRank** (`centrality.PageRank`) is a hybrid, and representative of real-world GraphBLAS code. The expensive step, pulling rank along in-links, is a plus-times mxv against the transposed graph, so it rides the sparse kernels. The bookkeeping around it (dividing rank by out-degree, collecting dangling mass, applying damping and teleport, measuring the L1 delta) runs as plain loops over dense vectors. Each of those could be dressed up as element-wise algebra; the loops are clearer and cost O(n) against the multiply's O(edges).

**Point-to-point shortest path** (`shortestpath.Between`) is deliberately not matrix algebra: it is Dijkstra with a binary heap and early exit. A priority queue settles one vertex at a time in a data-dependent order, which has no useful matrix formulation, and the early exit (stop the moment the target settles) is the entire point of a point-to-point query. Bulk semiring iterations cannot stop early that way. This mirrors the wider ecosystem: even SuiteSparse-based stacks step outside the algebra for this query shape, and LAGraph's SSSP uses delta-stepping rather than Dijkstra precisely because Dijkstra will not vectorize.

**The stubs** (`centrality` betweenness and closeness, `clustering`, all-pairs and temporal shortest paths) are unimplemented, but all of them are matrix-native on paper, which makes them approachable first contributions: betweenness is BFS-style forward sweeps plus a backward accumulation, both mxv-shaped; Markov clustering alternates mxm (expansion) with element-wise powers and rescaling (inflation); all-pairs shortest paths is repeated min-plus multiplication, `A^(n-1)` over the tropical semiring.

## Where the algebra stops: performance and hardware

The algebra is the specification, never the implementation. A min-plus multiply is *defined* as a matrix product but *executed* as a walk over `rowStart`/`cols`/`values`. Every operation in this library lands on one of three execution tiers, and "expressed as matrix algebra" says nothing about which:

| Operation | Default build | `GOEXPERIMENT=simd` build |
|---|---|---|
| dense x dense mxm and add, `float64`, unmasked | scalar axpy/add kernels over flat rows | vector kernels via Go 1.27's portable `simd` package (128-bit NEON on arm64, up to 512-bit AVX on amd64) |
| CSR/CSC mxv, plus-times | scalar walk over the compressed arrays | same, scalar |
| CSR/CSC mxv, any semiring | scalar walk plus two interface calls per stored entry | same, scalar |
| masked mxm, mixed operand types, everything else | generic interface-driven path | same |

Three findings from benchmarking this fork put the tiers in perspective (numbers and hardware details in the [README](../README.md)):

- **The big wins are kernel structure and memory layout, never lanes.** Moving dense mxm from the generic path to the scalar kernel was ~30x; moving sparse mxv was ~14,000x. Those gains come from contiguous access, allocation removal, and walking only stored entries. SIMD sits on top of that and is worth at most a small integer factor.
- **SIMD helps only where the data is dense and the lanes are wide.** The dense kernels' SIMD variants lose to the scalar kernels on 128-bit NEON for `float64` (two lanes cannot pay for the per-instruction overhead, and Go's arm64 compiler already fuses the scalar loop into FMA) and win about 2.6x for `float64` and 5.3x for `float32` on AVX-512, where eight and sixteen lanes amortize it. The SIMD build is opt-in precisely because the answer is hardware-dependent.
- **The sparse kernels, which carry every graph algorithm in this repo, cannot vectorize portably today.** Their inner loop is an index-directed gather (`x[cols[j]]`), and Go 1.27's `simd` packages ship no gather/scatter on any architecture. So BFS, PageRank, and shortest paths all run on scalar kernels regardless of build flags; the matrix-algebra framing buys them the fast *scalar* kernels, and would buy vectorization the day the toolchain grows gathers. Dense-path consumers, such as Strassen's base-case multiplies, are the ones that reach the SIMD tier now.

The pluggable semiring adds one more tier boundary: its two interface-method calls per stored entry cost roughly an order of magnitude against the hardcoded plus-times kernel (about 1.2 ms versus 75 µs for an mxv over 100k edges), while remaining orders of magnitude under the generic path. If that gap ever matters, monomorphized kernels for the stock semirings are the known fix.

## Further reading

- [The GraphBLAS Forum](https://graphblas.github.io/) and the [GraphBLAS C API specification](https://graphblas.org/)
- [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS), the reference implementation, and [LAGraph](https://github.com/GraphBLAS/LAGraph), the algorithm library on top of it
- Kepner & Gilbert, *Graph Algorithms in the Language of Linear Algebra* (SIAM, 2011)
- The runnable `Example` tests in this repo (`breadthfirst`, `centrality`, `shortestPath`, and the root package), which show each idea on a small named-vertex graph with verified output
