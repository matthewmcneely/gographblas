// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package shortestpath_test

import (
	"context"
	"math"
	"testing"

	"github.com/rossmerr/graphblas"
	shortestpath "github.com/rossmerr/graphblas/shortestPath"
)

// weights has shortest paths from vertex 0 of [0, 1, 3, 4] and leaves
// vertex 4 unreachable.
var weights = [][]float64{
	{0, 1, 4, 0, 0},
	{0, 0, 2, 5, 0},
	{0, 0, 0, 1, 0},
	{0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0},
}

func TestSingleSourceKnownDistances(t *testing.T) {
	want := []float64{0, 1, 3, 4}

	matrices := map[string]graphblas.Matrix[float64]{
		"dense": graphblas.NewDenseMatrixFromArrayN(weights),
		"csr":   graphblas.NewCSRMatrixFromArray(weights),
	}

	for name, m := range matrices {
		dist := shortestpath.SingleSource[float64](context.Background(), m, 0)

		for i, w := range want {
			if dist.AtVec(i) != w {
				t.Fatalf("%s: dist[%d] = %v, want %v", name, i, dist.AtVec(i), w)
			}
		}

		if !math.IsInf(dist.AtVec(4), 1) {
			t.Fatalf("%s: dist[4] = %v, want +Inf", name, dist.AtVec(4))
		}
	}
}

func TestSingleSourceNegativeEdges(t *testing.T) {
	// A DAG with a negative edge: the cheapest route to vertex 2 goes
	// through vertex 1.
	m := graphblas.NewDenseMatrixFromArrayN([][]float64{
		{0, 2, 5, 0},
		{0, 0, -1, 0},
		{0, 0, 0, 2},
		{0, 0, 0, 0},
	})

	dist := shortestpath.SingleSource[float64](context.Background(), m, 0)

	want := []float64{0, 2, 1, 3}
	for i, w := range want {
		if dist.AtVec(i) != w {
			t.Fatalf("dist[%d] = %v, want %v", i, dist.AtVec(i), w)
		}
	}
}

func TestSingleSourceFromOtherVertex(t *testing.T) {
	m := graphblas.NewDenseMatrixFromArrayN(weights)

	dist := shortestpath.SingleSource[float64](context.Background(), m, 1)

	if dist.AtVec(1) != 0 {
		t.Fatalf("dist[1] = %v, want 0", dist.AtVec(1))
	}
	if dist.AtVec(3) != 3 {
		t.Fatalf("dist[3] = %v, want 3", dist.AtVec(3))
	}
	if !math.IsInf(dist.AtVec(0), 1) {
		t.Fatalf("dist[0] = %v, want +Inf", dist.AtVec(0))
	}
}

func TestSingleSourceFloat32(t *testing.T) {
	m := graphblas.NewDenseMatrixFromArrayN([][]float32{
		{0, 1.5, 0},
		{0, 0, 2.5},
		{0, 0, 0},
	})

	dist := shortestpath.SingleSource[float32](context.Background(), m, 0)

	if dist.AtVec(2) != 4 {
		t.Fatalf("dist[2] = %v, want 4", dist.AtVec(2))
	}
}
