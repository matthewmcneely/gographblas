// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering

import (
	"context"
	"log"
	"math"

	"github.com/matthewmcneely/gographblas"
	"github.com/matthewmcneely/gographblas/constraints"
)

const (
	// DefaultInflation is the exponent Markov raises flow values to after
	// each expansion. Larger values separate flow faster and yield more,
	// smaller clusters.
	DefaultInflation = 2
	// DefaultMarkovIterations caps the expansion and inflation rounds when
	// the flow has not converged.
	DefaultMarkovIterations = 100
)

const (
	// markovPrune drops flow values below this share of their row after
	// inflation, which keeps the iterate sparse.
	markovPrune = 1e-5
	// markovTolerance is the largest change in any flow value under which
	// the iteration is considered converged.
	markovTolerance = 1e-6
)

// Markov clusters the vertices of a by Markov clustering (MCL): random
// walks on the graph are simulated by repeatedly squaring a stochastic
// flow matrix (expansion, one sparse mxm per round), and raising its
// values to the DefaultInflation power then rescaling (inflation), which
// strengthens flow inside dense regions and starves it between them. The
// graph is treated as undirected, with a.At(u, v) the weight of edge
// u - v and zero marking an absent edge; weights must be non-negative.
// Every vertex gets a self-loop weighted like its heaviest edge, as MCL
// prescribes. Clusters are the connected components of the converged flow
// and are labeled as the package documentation describes.
func Markov[T constraints.Float](ctx context.Context, a graphblas.Matrix[T]) []int {
	return MarkovWithOptions[T](ctx, a, DefaultInflation, DefaultMarkovIterations)
}

// MarkovWithOptions is Markov with an explicit inflation exponent, which
// must exceed 1, and iteration cap.
func MarkovWithOptions[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], inflation T, maxIterations int) []int {
	if !(inflation > 1) {
		log.Panicf("Markov inflation must exceed 1, found %+v", inflation)
	}

	w, _ := symmetrize("Markov", a)
	n := w.Rows()

	heaviest := make([]T, n)
	var edges []graphblas.Edge[T]
	for iterator := w.Enumerate(); iterator.HasNext(); {
		u, v, weight := iterator.Next()
		heaviest[u] = max(heaviest[u], weight)
		edges = append(edges, graphblas.Edge[T]{From: u, To: v, Weight: weight})
	}
	for u, h := range heaviest {
		if h == 0 {
			h = 1
		}
		edges = append(edges, graphblas.Edge[T]{From: u, To: u, Weight: h})
	}

	// Row-stochastic flow: row u holds where a walk at u moves next. A
	// power of 1 normalizes without inflating.
	flow := inflate(graphblas.NewCSRMatrixFromEdges(n, n, edges), 1)
	m := graphblas.NewCSRMatrixFromEdges(n, n, flow)

	for iteration := 0; iteration < maxIterations; iteration++ {
		select {
		case <-ctx.Done():
			return markovClusters(m)
		default:
		}

		expanded := graphblas.NewCSRMatrix[T](n, n)
		graphblas.MatrixMatrixMultiply[T](ctx, m, m, nil, expanded)

		next := inflate(expanded, inflation)
		m = graphblas.NewCSRMatrixFromEdges(n, n, next)

		converged := largestChange(flow, next) < markovTolerance
		flow = next
		if converged {
			break
		}
	}

	return markovClusters(m)
}

// inflate raises every stored value of the row-stochastic matrix m to the
// given power, rescales each row to sum to 1, prunes values below
// markovPrune of their row (always keeping the row's largest), and
// rescales the survivors. Entries come back in row-major order.
func inflate[T constraints.Float](m graphblas.Matrix[T], power T) []graphblas.Edge[T] {
	var out, row []graphblas.Edge[T]

	flush := func() {
		if len(row) == 0 {
			return
		}

		sum, largest := T(0), T(0)
		for i := range row {
			if power != 1 {
				row[i].Weight = T(math.Pow(float64(row[i].Weight), float64(power)))
			}
			sum += row[i].Weight
			largest = max(largest, row[i].Weight)
		}

		kept := T(0)
		start := len(out)
		for _, e := range row {
			if e.Weight == largest || e.Weight/sum >= markovPrune {
				out = append(out, e)
				kept += e.Weight
			}
		}
		for i := start; i < len(out); i++ {
			out[i].Weight /= kept
		}

		row = row[:0]
	}

	for iterator := m.Enumerate(); iterator.HasNext(); {
		u, v, weight := iterator.Next()
		if len(row) > 0 && row[0].From != u {
			flush()
		}
		row = append(row, graphblas.Edge[T]{From: u, To: v, Weight: weight})
	}
	flush()

	return out
}

// largestChange is the largest absolute difference between two matrices
// given as row-major entry lists, treating missing entries as zero.
func largestChange[T constraints.Float](a, b []graphblas.Edge[T]) T {
	var largest T
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		var d T
		switch {
		case j == len(b) || (i < len(a) && before(a[i], b[j])):
			d = a[i].Weight
			i++
		case i == len(a) || before(b[j], a[i]):
			d = b[j].Weight
			j++
		default:
			d = a[i].Weight - b[j].Weight
			i++
			j++
		}
		largest = max(largest, T(math.Abs(float64(d))))
	}
	return largest
}

func before[T constraints.Number](a, b graphblas.Edge[T]) bool {
	return a.From < b.From || (a.From == b.From && a.To < b.To)
}

// markovClusters labels the connected components of the flow matrix's
// pattern: each vertex joins the attractors its flow converged on.
func markovClusters[T constraints.Float](m graphblas.Matrix[T]) []int {
	parent := make([]int, m.Rows())
	for i := range parent {
		parent[i] = i
	}

	find := func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}

	for iterator := m.Enumerate(); iterator.HasNext(); {
		u, v, _ := iterator.Next()
		if ru, rv := find(u), find(v); ru != rv {
			parent[max(ru, rv)] = min(ru, rv)
		}
	}

	labels := make([]int, len(parent))
	for i := range labels {
		labels[i] = find(i)
	}
	return canonical(labels)
}
