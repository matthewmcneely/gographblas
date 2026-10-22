// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/rossmerr/graphblas"
)

// TestLeadingEigenvectorsSpanInvariantSubspace checks the eigensolver
// directly: the returned basis must be orthonormal, (I + N) / 2 must map
// it back into its own span, and the span must contain the known leading
// eigenvector D^1/2 1.
func TestLeadingEigenvectorsSpanInvariantSubspace(t *testing.T) {
	const n, k = 60, 3
	rnd := rand.New(rand.NewSource(4))

	var edges []graphblas.Edge[float64]
	for u := 0; u < n; u++ {
		for v := u + 1; v < n; v++ {
			p := 0.03
			if u/20 == v/20 {
				p = 0.4
			}
			if rnd.Float64() < p {
				w := 1 + rnd.Float64()
				edges = append(edges, graphblas.Edge[float64]{From: u, To: v, Weight: w}, graphblas.Edge[float64]{From: v, To: u, Weight: w})
			}
		}
	}
	w := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	degree := make([]float64, n)
	for iterator := w.Enumerate(); iterator.HasNext(); {
		u, _, weight := iterator.Next()
		degree[u] += weight
	}
	invSqrt := make([]float64, n)
	for i, d := range degree {
		if d > 0 {
			invSqrt[i] = 1 / math.Sqrt(d)
		}
	}

	apply := func(q []float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = q[i] / 2
		}
		for iterator := w.Enumerate(); iterator.HasNext(); {
			u, v, weight := iterator.Next()
			out[u] += invSqrt[u] * weight * invSqrt[v] * q[v] / 2
		}
		return out
	}

	basis := leadingEigenvectors(context.Background(), w, invSqrt, k, 1e-10, 10000)

	for i := range basis {
		for j := range basis {
			want := 0.0
			if i == j {
				want = 1
			}
			if got := dot(basis[i], basis[j]); math.Abs(got-want) > 1e-9 {
				t.Fatalf("<q%d, q%d> = %v, want %v", i, j, got, want)
			}
		}
	}

	for j, q := range basis {
		mq := apply(q)
		residual := slicesClone(mq)
		for _, p := range basis {
			d := dot(mq, p)
			for i := range residual {
				residual[i] -= d * p[i]
			}
		}
		if r := math.Sqrt(dot(residual, residual)); r > 1e-6 {
			t.Fatalf("basis vector %d leaves the subspace by %v", j, r)
		}
	}

	// The top eigenvalue of (I + N) / 2 is exactly 1 on a connected graph,
	// with D^1/2 1 as its eigenvector; it must lie in the span.
	top := make([]float64, n)
	for i, d := range degree {
		top[i] = math.Sqrt(d)
	}
	norm := math.Sqrt(dot(top, top))
	captured := 0.0
	for _, p := range basis {
		captured += math.Pow(dot(top, p)/norm, 2)
	}
	if math.Abs(captured-1) > 1e-9 {
		t.Fatalf("leading eigenvector captured %v of the way, want 1", captured)
	}
}

func slicesClone(s []float64) []float64 {
	return append([]float64(nil), s...)
}
