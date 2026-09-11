// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas_test

import (
	"context"
	"fmt"
	"math"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/binaryop"
)

// Swapping the semiring turns the same multiply into a different graph
// algorithm. Over (min, +), one matrix-vector multiply by the transposed
// graph advances shortest-path distances by a single edge relaxation: with
// only the source settled, one step yields the one-hop distances.
func ExampleMatrixVectorMultiplyWithSemiring() {
	const (
		a = iota
		b
		c
		numVertices
	)

	g := graphblas.NewCSRMatrixFromEdges(numVertices, numVertices, []graphblas.Edge[float64]{
		{From: a, To: b, Weight: 5},
		{From: a, To: c, Weight: 2},
		{From: b, To: c, Weight: 1},
	})

	ctx := context.Background()
	at := graphblas.TransposeToCSR[float64](ctx, g)

	dist := graphblas.NewDenseVectorN[float64](numVertices)
	dist.SetVec(a, 0)
	dist.SetVec(b, math.Inf(1))
	dist.SetVec(c, math.Inf(1))

	next := graphblas.NewDenseVectorN[float64](numVertices)
	graphblas.MatrixVectorMultiplyWithSemiring[float64](ctx, at, dist, nil, next, binaryop.MinPlus[float64]())

	fmt.Println("one hop from a:", next.AtVec(a), next.AtVec(b), next.AtVec(c))
	// Output:
	// one hop from a: +Inf 5 2
}
