// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering

import (
	"cmp"
	"context"
	"log"
	"math"
	"slices"

	"github.com/matthewmcneely/gographblas"
	"github.com/matthewmcneely/gographblas/constraints"
)

const (
	// DefaultTeleport is the probability the personalized random walk
	// behind Local jumps back to the seed. Smaller values let the walk
	// spread further and find larger clusters.
	DefaultTeleport = 0.15
	// DefaultEpsilon is the residual per unit of degree below which Local
	// stops pushing probability from a vertex. Smaller values explore
	// further at more cost.
	DefaultEpsilon = 1e-6
)

// Local finds the cluster around a seed vertex of a, using the
// Andersen-Chung-Lang method: approximate the personalized PageRank
// vector of the seed by pushing probability outward from it, then sweep
// the vertices in decreasing order of PageRank per unit of degree and
// return the prefix with the lowest conductance (the weight of edges
// leaving the set over the smaller of its volume and the rest of the
// graph's). Pushes run in parallel rounds, each one sparse mxv that
// spreads the residual of every vertex still above the threshold, as in
// Shun et al.'s parallel formulation. The graph is treated as undirected,
// with a.At(u, v) the weight of edge u - v and zero marking an absent
// edge; weights must be non-negative. Conductance favors larger sets up
// to half the graph's volume, so the cluster can take in neighboring
// communities that are about as well separated from the rest; raising
// the teleport probability or epsilon keeps it closer to the seed (the
// push reaches a volume of at most 1/(teleport*epsilon)), though not
// monotonically, since the sweep is approximate. Uses DefaultTeleport and
// DefaultEpsilon. Exact personalized PageRank per unit of degree peaks
// at the seed, so the seed leads the sweep; only prefixes containing it
// are candidates, which keeps that a guarantee under the approximation.
// The members come back in vertex order, and an isolated seed is a
// cluster of one.
func Local[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], seed int) []int {
	return LocalWithOptions[T](ctx, a, seed, DefaultTeleport, DefaultEpsilon)
}

// LocalWithOptions is Local with an explicit teleport probability in
// (0, 1) and push threshold epsilon.
func LocalWithOptions[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], seed int, teleport, epsilon T) []int {
	if !(teleport > 0 && teleport < 1) {
		log.Panicf("Local teleport must lie in (0, 1), found %+v", teleport)
	}

	if !(epsilon > 0) {
		log.Panicf("Local epsilon must be positive, found %+v", epsilon)
	}

	w, degree := symmetrize("Local", a)
	n := w.Rows()
	if seed < 0 || seed >= n {
		log.Panicf("Seed '%+v' is invalid", seed)
	}

	if degree[seed] == 0 {
		return []int{seed}
	}

	pagerank := make([]T, n)
	residual := make([]T, n)
	residual[seed] = 1

	xData := make([]T, n)
	x := graphblas.NewDenseVectorFromArrayN(xData)
	yData := make([]T, n)
	y := graphblas.NewDenseVectorFromArrayN(yData)

	// Each push keeps teleport of a vertex's residual as PageRank, holds
	// half the rest at the vertex (the lazy walk), and spreads the other
	// half across its edges in proportion to weight.
	for {
		select {
		case <-ctx.Done():
			return sweep(w, degree, pagerank, seed)
		default:
		}

		active := false
		for u, r := range residual {
			xData[u] = 0
			if r > 0 && r >= epsilon*degree[u] {
				xData[u] = r / degree[u]
				pagerank[u] += teleport * r
				residual[u] = (1 - teleport) * r / 2
				active = true
			}
		}
		if !active {
			break
		}

		graphblas.MatrixVectorMultiply[T](ctx, w, x, nil, y)

		for v, spread := range yData {
			residual[v] += (1 - teleport) / 2 * spread
		}
	}

	return sweep(w, degree, pagerank, seed)
}

// sweep orders the vertices holding PageRank by PageRank per unit of
// degree and returns, in vertex order, the prefix containing the seed
// with the lowest conductance. Adding vertex v to the set turns its edges into the set
// from cut edges into internal ones and makes its other edges, self-loops
// aside, cut edges.
func sweep[T constraints.Float](w graphblas.Matrix[T], degree, pagerank []T, seed int) []int {
	n := len(degree)

	neighbors := make([][]graphblas.Edge[T], n)
	for iterator := w.Enumerate(); iterator.HasNext(); {
		u, v, weight := iterator.Next()
		neighbors[u] = append(neighbors[u], graphblas.Edge[T]{From: u, To: v, Weight: weight})
	}

	var order []int
	var total T
	for v, p := range pagerank {
		total += degree[v]
		if p > 0 {
			order = append(order, v)
		}
	}
	slices.SortStableFunc(order, func(u, v int) int {
		return cmp.Compare(pagerank[v]/degree[v], pagerank[u]/degree[u])
	})

	inSet := make([]bool, n)
	var volume, cut T
	best, bestConductance := -1, T(math.Inf(1))
	for i, v := range order {
		var toSet, loop T
		for _, e := range neighbors[v] {
			switch {
			case e.To == v:
				loop += e.Weight
			case inSet[e.To]:
				toSet += e.Weight
			}
		}

		cut += degree[v] - loop - 2*toSet
		volume += degree[v]
		inSet[v] = true

		if smaller := min(volume, total-volume); inSet[seed] && smaller > 0 {
			if conductance := cut / smaller; conductance < bestConductance {
				best, bestConductance = i, conductance
			}
		}
	}

	if best < 0 {
		return []int{seed}
	}

	members := slices.Clone(order[:best+1])
	slices.Sort(members)
	return members
}
