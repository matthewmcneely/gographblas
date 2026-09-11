// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package breadthfirst_test

import (
	"context"
	"fmt"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/breadthfirst"
)

// ExampleSearch counts how many hops separate two people in a follow
// graph. Search advances the frontier along matrix columns, so the graph
// is transposed once up front to traverse in edge direction.
func ExampleSearch() {
	const (
		alice = iota
		bob
		carol
		dave
		erin
		numPeople
	)

	follows := graphblas.NewCSRMatrixFromEdges(numPeople, numPeople, []graphblas.Edge[float64]{
		{From: alice, To: bob, Weight: 1},
		{From: alice, To: dave, Weight: 1},
		{From: bob, To: carol, Weight: 1},
		{From: carol, To: erin, Weight: 1},
	})

	ctx := context.Background()
	reach := graphblas.TransposeToCSR[float64](ctx, follows)

	hops := 0
	breadthfirst.Search[float64](ctx, reach, alice, func(frontier graphblas.Vector[float64]) bool {
		hops++
		return frontier.AtVec(erin) > 0
	})

	fmt.Printf("erin is %d hops from alice\n", hops)
	// Output:
	// erin is 3 hops from alice
}
