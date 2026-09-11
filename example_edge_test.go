// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package graphblas_test

import (
	"fmt"

	"github.com/rossmerr/graphblas"
)

// Building a graph from an edge list reads better than a full 2D array:
// each line is one edge, and absent edges are simply not written.
func ExampleNewCSRMatrixFromEdges() {
	g := graphblas.NewCSRMatrixFromEdges(3, 3, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 2},
		{From: 0, To: 2, Weight: 3},
		{From: 2, To: 1, Weight: 4},
	})

	fmt.Println(g.At(0, 1), g.At(0, 2), g.At(2, 1), g.At(1, 0))
	fmt.Println("stored entries:", g.Values())
	// Output:
	// 2 3 4 0
	// stored entries: 3
}
