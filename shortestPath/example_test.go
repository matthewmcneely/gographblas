// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package shortestpath_test

import (
	"context"
	"fmt"

	"github.com/rossmerr/graphblas"
	shortestpath "github.com/rossmerr/graphblas/shortestPath"
)

// ExampleSingleSource finds driving distances from Seattle on a small road
// map. Going through Portland to Boise costs 174 + 430 = 604, so the
// direct 496 road wins; Chicago has no incoming roads and reports +Inf.
func ExampleSingleSource() {
	const (
		seattle = iota
		portland
		boise
		denver
		chicago
		numCities
	)

	roads := graphblas.NewCSRMatrixFromEdges(numCities, numCities, []graphblas.Edge[float64]{
		{From: seattle, To: portland, Weight: 174},
		{From: seattle, To: boise, Weight: 496},
		{From: portland, To: boise, Weight: 430},
		{From: boise, To: denver, Weight: 830},
	})

	dist := shortestpath.SingleSource[float64](context.Background(), roads, seattle)

	for i, name := range []string{"seattle", "portland", "boise", "denver", "chicago"} {
		fmt.Printf("%s: %v\n", name, dist.AtVec(i))
	}
	// Output:
	// seattle: 0
	// portland: 174
	// boise: 496
	// denver: 1326
	// chicago: +Inf
}
