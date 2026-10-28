// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering

import (
	"context"
	"log"
	"math"
	"math/rand/v2"

	"github.com/matthewmcneely/gographblas"
	"github.com/matthewmcneely/gographblas/constraints"
)

const (
	// DefaultSpectralTolerance is the subspace change under which the
	// eigenvector iteration is considered converged.
	DefaultSpectralTolerance = 1e-6
	// DefaultSpectralIterations caps the eigenvector iteration when the
	// tolerance is not reached.
	DefaultSpectralIterations = 1000
)

// kmeansIterations caps Lloyd's algorithm on the spectral embedding.
const kmeansIterations = 100

// Spectral partitions the vertices of a into at most k clusters by
// normalized spectral clustering (Ng, Jordan and Weiss): it embeds each
// vertex as its row of the k leading eigenvectors of the normalized
// adjacency D^-1/2 W D^-1/2 (the k smallest of the normalized Laplacian),
// normalizes the rows, and groups them with k-means. The graph is treated
// as undirected, with a.At(u, v) the weight of edge u - v and zero
// marking an absent edge; weights must be non-negative. The eigenvectors
// come from orthogonal iteration, k sparse mxv per round, computed in
// float64 whatever T is. The iteration, starting vectors and k-means
// seeding are deterministic, so equal inputs give equal labels. Labels
// are numbered as the package documentation describes; fewer than k
// clusters come back when the embedding has fewer than k distinct
// points.
func Spectral[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], k int) []int {
	return SpectralWithOptions[T](ctx, a, k, DefaultSpectralTolerance, DefaultSpectralIterations)
}

// SpectralWithOptions is Spectral with an explicit subspace convergence
// tolerance and iteration cap for the eigenvector computation.
func SpectralWithOptions[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], k int, tolerance float64, maxIterations int) []int {
	w, degree := symmetrize("Spectral", a)
	n := w.Rows()
	if n == 0 {
		return []int{}
	}

	if k < 1 || k > n {
		log.Panicf("Spectral requires 1 <= k <= %+v, found %+v", n, k)
	}

	if k == 1 {
		return make([]int, n)
	}

	var edges []graphblas.Edge[float64]
	for iterator := w.Enumerate(); iterator.HasNext(); {
		u, v, weight := iterator.Next()
		edges = append(edges, graphblas.Edge[float64]{From: u, To: v, Weight: float64(weight)})
	}
	w64 := graphblas.NewCSRMatrixFromEdges(n, n, edges)

	invSqrt := make([]float64, n)
	for i, d := range degree {
		if d > 0 {
			invSqrt[i] = 1 / math.Sqrt(float64(d))
		}
	}

	basis := leadingEigenvectors(ctx, w64, invSqrt, k, tolerance, maxIterations)

	points := make([][]float64, n)
	for i := range points {
		p := make([]float64, k)
		norm := 0.0
		for j := range basis {
			p[j] = basis[j][i]
			norm += p[j] * p[j]
		}
		if norm > 0 {
			norm = math.Sqrt(norm)
			for j := range p {
				p[j] /= norm
			}
		}
		points[i] = p
	}

	return canonical(kmeans(points, k))
}

// leadingEigenvectors returns an orthonormal basis for the k leading
// eigenvectors of the normalized adjacency N = D^-1/2 W D^-1/2 by
// orthogonal iteration on (I + N) / 2, whose eigenvalues lie in [0, 1]
// in the same order as N's, so the leading ones dominate. An isolated
// vertex is its own component, so N gets a 1 on its diagonal, matching
// the normalized Laplacian's zero eigenvalue per component.
func leadingEigenvectors(ctx context.Context, w graphblas.Matrix[float64], invSqrt []float64, k int, tolerance float64, maxIterations int) [][]float64 {
	n := len(invSqrt)
	rnd := rand.New(rand.NewPCG(1, 2))

	// Start from the known leading eigenvector D^1/2 1 and random
	// vectors, so the start is never deficient in a leading direction.
	basis := make([][]float64, k)
	for j := range basis {
		basis[j] = make([]float64, n)
		for i := range basis[j] {
			if j == 0 && invSqrt[i] > 0 {
				basis[j][i] = 1 / invSqrt[i]
			} else if j > 0 {
				basis[j][i] = rnd.NormFloat64()
			}
		}
	}
	orthonormalize(basis, rnd)

	xData := make([]float64, n)
	x := graphblas.NewDenseVectorFromArrayN(xData)
	yData := make([]float64, n)
	y := graphblas.NewDenseVectorFromArrayN(yData)

	next := make([][]float64, k)
	for j := range next {
		next[j] = make([]float64, n)
	}

	for iteration := 0; iteration < maxIterations; iteration++ {
		select {
		case <-ctx.Done():
			return basis
		default:
		}

		for j, q := range basis {
			for i := range xData {
				xData[i] = q[i] * invSqrt[i]
			}

			graphblas.MatrixVectorMultiply[float64](ctx, w, x, nil, y)

			for i := range yData {
				if invSqrt[i] == 0 {
					next[j][i] = q[i]
				} else {
					next[j][i] = (q[i] + invSqrt[i]*yData[i]) / 2
				}
			}
		}
		orthonormalize(next, rnd)

		// The residual of the new basis projected onto the old one
		// measures how far the subspace moved.
		change := 0.0
		for _, v := range next {
			norm := 0.0
			for _, q := range basis {
				norm += dot(v, q) * dot(v, q)
			}
			change += max(0, 1-norm)
		}

		basis, next = next, basis
		if math.Sqrt(change) < tolerance {
			break
		}
	}

	return basis
}

// orthonormalize applies modified Gram-Schmidt to the vectors in place,
// replacing any vector that collapses into the span of the earlier ones
// with a fresh random direction. There are never more vectors than
// dimensions, so a random direction always survives.
func orthonormalize(vectors [][]float64, rnd *rand.Rand) {
	for j, v := range vectors {
		for {
			for _, q := range vectors[:j] {
				d := dot(v, q)
				for i := range v {
					v[i] -= d * q[i]
				}
			}

			norm := math.Sqrt(dot(v, v))
			if norm > 1e-10 {
				for i := range v {
					v[i] /= norm
				}
				break
			}

			for i := range v {
				v[i] = rnd.NormFloat64()
			}
		}
	}
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// kmeans groups the points into at most k clusters with Lloyd's
// algorithm, seeded by farthest-first traversal from the first point.
func kmeans(points [][]float64, k int) []int {
	n := len(points)

	centers := [][]float64{append([]float64(nil), points[0]...)}
	nearest := make([]float64, n)
	for i, p := range points {
		nearest[i] = distance(p, centers[0])
	}
	for len(centers) < k {
		far := 0
		for i := range points {
			if nearest[i] > nearest[far] {
				far = i
			}
		}
		if nearest[far] == 0 {
			break
		}

		c := append([]float64(nil), points[far]...)
		centers = append(centers, c)
		for i, p := range points {
			nearest[i] = min(nearest[i], distance(p, c))
		}
	}

	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}

	for iteration := 0; iteration < kmeansIterations; iteration++ {
		moved := false
		for i, p := range points {
			best := 0
			for c := 1; c < len(centers); c++ {
				if distance(p, centers[c]) < distance(p, centers[best]) {
					best = c
				}
			}
			if labels[i] != best {
				labels[i] = best
				moved = true
			}
		}
		if !moved {
			break
		}

		// An emptied cluster keeps its previous center.
		counts := make([]int, len(centers))
		sums := make([][]float64, len(centers))
		for c := range sums {
			sums[c] = make([]float64, len(centers[c]))
		}
		for i, p := range points {
			counts[labels[i]]++
			for j, v := range p {
				sums[labels[i]][j] += v
			}
		}
		for c, count := range counts {
			if count == 0 {
				continue
			}
			for j := range centers[c] {
				centers[c][j] = sums[c][j] / float64(count)
			}
		}
	}

	return labels
}

// distance is the squared Euclidean distance between two points.
func distance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
