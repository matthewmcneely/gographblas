// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package shortestpath

import (
	"container/heap"
	"context"
	"log"
	"math"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/constraints"
)

// Between returns the shortest distance from vertex s to vertex t and the
// path that achieves it, as vertex indices from s to t inclusive. It runs
// Dijkstra's algorithm and stops as soon as t is settled, so a query
// explores only the region of the graph nearer than t. Edge weights must
// be non-negative; use SingleSource when negative weights are involved.
// a.At(u, v) holds the weight of edge u->v and zero marks an absent edge.
// When t is unreachable, or ctx is cancelled before t settles, the
// distance is +Inf and the path is nil.
func Between[T constraints.Float](ctx context.Context, a graphblas.Matrix[T], s, t int) (T, []int) {
	n := a.Rows()
	if a.Columns() != n {
		log.Panicf("Between requires a square matrix, found %+v x %+v", n, a.Columns())
	}

	if s < 0 || s >= n {
		log.Panicf("Source '%+v' is invalid", s)
	}

	if t < 0 || t >= n {
		log.Panicf("Target '%+v' is invalid", t)
	}

	if s == t {
		return 0, []int{s}
	}

	type arc struct {
		to     int
		weight T
	}

	adjacency := make([][]arc, n)
	for iterator := a.Enumerate(); iterator.HasNext(); {
		u, v, w := iterator.Next()
		if w == 0 {
			continue
		}

		if w < 0 {
			log.Panicf("Between requires non-negative edge weights, found '%+v' on edge %+v -> %+v", w, u, v)
		}

		adjacency[u] = append(adjacency[u], arc{to: v, weight: w})
	}

	inf := T(math.Inf(1))
	dist := make([]T, n)
	pred := make([]int, n)
	settled := make([]bool, n)
	for i := 0; i < n; i++ {
		dist[i] = inf
		pred[i] = -1
	}
	dist[s] = 0

	pq := &pqHeap[T]{{vertex: s, dist: 0}}
	for pops := 0; pq.Len() > 0; pops++ {
		if pops&1023 == 0 {
			select {
			case <-ctx.Done():
				return inf, nil
			default:
			}
		}

		item := heap.Pop(pq).(pqItem[T])
		u := item.vertex
		if settled[u] {
			continue
		}
		settled[u] = true

		if u == t {
			break
		}

		for _, e := range adjacency[u] {
			if d := item.dist + e.weight; d < dist[e.to] {
				dist[e.to] = d
				pred[e.to] = u
				heap.Push(pq, pqItem[T]{vertex: e.to, dist: d})
			}
		}
	}

	if !settled[t] {
		return inf, nil
	}

	path := make([]int, 0, 8)
	for v := t; v != -1; v = pred[v] {
		path = append(path, v)
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	return dist[t], path
}

type pqItem[T constraints.Float] struct {
	vertex int
	dist   T
}

type pqHeap[T constraints.Float] []pqItem[T]

func (h pqHeap[T]) Len() int           { return len(h) }
func (h pqHeap[T]) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h pqHeap[T]) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *pqHeap[T]) Push(x any)        { *h = append(*h, x.(pqItem[T])) }
func (h *pqHeap[T]) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
