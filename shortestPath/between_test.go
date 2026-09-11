// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package shortestpath_test

import (
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/rossmerr/graphblas"
	shortestpath "github.com/rossmerr/graphblas/shortestPath"
)

func TestBetweenKnownPath(t *testing.T) {
	// The unique shortest route 0 -> 3 is 0 -> 1 -> 2 -> 3 with cost 4;
	// the direct 0 -> 2 road costs 4 so going through 1 (cost 3) wins.
	matrices := map[string]graphblas.Matrix[float64]{
		"dense": graphblas.NewDenseMatrixFromArrayN(weights),
		"csr":   graphblas.NewCSRMatrixFromArray(weights),
	}

	for name, m := range matrices {
		dist, path := shortestpath.Between[float64](context.Background(), m, 0, 3)

		if dist != 4 {
			t.Fatalf("%s: dist = %v, want 4", name, dist)
		}
		if want := []int{0, 1, 2, 3}; !reflect.DeepEqual(path, want) {
			t.Fatalf("%s: path = %v, want %v", name, path, want)
		}
	}
}

func TestBetweenUnreachable(t *testing.T) {
	m := graphblas.NewCSRMatrixFromArray(weights)

	dist, path := shortestpath.Between[float64](context.Background(), m, 0, 4)

	if !math.IsInf(dist, 1) {
		t.Fatalf("dist = %v, want +Inf", dist)
	}
	if path != nil {
		t.Fatalf("path = %v, want nil", path)
	}
}

func TestBetweenSameVertex(t *testing.T) {
	m := graphblas.NewCSRMatrixFromArray(weights)

	dist, path := shortestpath.Between[float64](context.Background(), m, 2, 2)

	if dist != 0 {
		t.Fatalf("dist = %v, want 0", dist)
	}
	if want := []int{2}; !reflect.DeepEqual(path, want) {
		t.Fatalf("path = %v, want %v", path, want)
	}
}

func TestBetweenAgreesWithSingleSource(t *testing.T) {
	// Distances from Between must match SingleSource on a graph with
	// non-negative weights, for every reachable target.
	m := graphblas.NewCSRMatrixFromEdges(6, 6, []graphblas.Edge[float64]{
		{From: 0, To: 1, Weight: 7},
		{From: 0, To: 2, Weight: 9},
		{From: 0, To: 5, Weight: 14},
		{From: 1, To: 2, Weight: 10},
		{From: 1, To: 3, Weight: 15},
		{From: 2, To: 3, Weight: 11},
		{From: 2, To: 5, Weight: 2},
		{From: 3, To: 4, Weight: 6},
		{From: 5, To: 4, Weight: 9},
	})

	want := shortestpath.SingleSource[float64](context.Background(), m, 0)

	for target := 0; target < 6; target++ {
		dist, path := shortestpath.Between[float64](context.Background(), m, 0, target)

		if dist != want.AtVec(target) {
			t.Fatalf("dist to %d = %v, want %v", target, dist, want.AtVec(target))
		}

		// The path must start at the source, end at the target, and its
		// edge weights must sum to the reported distance.
		if len(path) == 0 || path[0] != 0 || path[len(path)-1] != target {
			t.Fatalf("path to %d = %v, want a path from 0 to %d", target, path, target)
		}

		sum := 0.0
		for i := 1; i < len(path); i++ {
			w := m.At(path[i-1], path[i])
			if w == 0 && path[i-1] != path[i] {
				t.Fatalf("path to %d uses absent edge %d -> %d", target, path[i-1], path[i])
			}
			sum += w
		}
		if sum != dist {
			t.Fatalf("path to %d sums to %v, want %v", target, sum, dist)
		}
	}
}

func TestBetweenFloat32(t *testing.T) {
	m := graphblas.NewCSRMatrixFromEdges(3, 3, []graphblas.Edge[float32]{
		{From: 0, To: 1, Weight: 1.5},
		{From: 1, To: 2, Weight: 2.5},
	})

	dist, path := shortestpath.Between[float32](context.Background(), m, 0, 2)

	if dist != 4 {
		t.Fatalf("dist = %v, want 4", dist)
	}
	if want := []int{0, 1, 2}; !reflect.DeepEqual(path, want) {
		t.Fatalf("path = %v, want %v", path, want)
	}
}
