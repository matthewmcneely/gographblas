// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package clustering_test

import (
	"context"
	"fmt"

	"github.com/matthewmcneely/gographblas"
	"github.com/matthewmcneely/gographblas/clustering"
)

// friendships is a small social network: a book club and a climbing gym
// whose members all know each other, joined only by Dan (book club)
// knowing Eve (climbing). Each friendship is listed once; the undirected
// algorithms read both directions from it.
func friendships() ([]string, []graphblas.Edge[float64]) {
	names := []string{"ana", "ben", "cat", "dan", "eve", "fay", "gus", "hal"}
	var edges []graphblas.Edge[float64]
	for _, group := range [][]int{{0, 1, 2, 3}, {4, 5, 6, 7}} {
		for i, u := range group {
			for _, v := range group[i+1:] {
				edges = append(edges, graphblas.Edge[float64]{From: u, To: v, Weight: 1})
			}
		}
	}
	edges = append(edges, graphblas.Edge[float64]{From: 3, To: 4, Weight: 1})
	return names, edges
}

func printGroups(names []string, labels []int) {
	for i, members := range clustering.Groups(labels) {
		fmt.Print(i, ":")
		for _, v := range members {
			fmt.Print(" ", names[v])
		}
		fmt.Println()
	}
}

// ExampleMarkov separates the book club from the climbing gym: random
// walks rarely cross the single Dan-Eve friendship, and inflation cuts the
// flow across it.
func ExampleMarkov() {
	names, edges := friendships()
	g := graphblas.NewCSRMatrixFromEdges(len(names), len(names), edges)

	printGroups(names, clustering.Markov[float64](context.Background(), g))
	// Output:
	// 0: ana ben cat dan
	// 1: eve fay gus hal
}

// ExamplePeerPressure places a newcomer by who follows whom. Kim is
// followed by the whole design team and follows the whole data team.
// Followers vote for the accounts they follow to join their cluster, so
// Kim lands with design.
func ExamplePeerPressure() {
	const (
		ava = iota // design
		bo
		cy
		dee // data
		eli
		flo
		kim
		numPeople
	)

	var follows []graphblas.Edge[float64]
	for _, team := range [][]int{{ava, bo, cy}, {dee, eli, flo}} {
		for _, u := range team {
			for _, v := range team {
				if u != v {
					follows = append(follows, graphblas.Edge[float64]{From: u, To: v, Weight: 1})
				}
			}
		}
	}
	for _, designer := range []int{ava, bo, cy} {
		follows = append(follows, graphblas.Edge[float64]{From: designer, To: kim, Weight: 1})
	}
	for _, analyst := range []int{dee, eli, flo} {
		follows = append(follows, graphblas.Edge[float64]{From: kim, To: analyst, Weight: 1})
	}

	g := graphblas.NewCSRMatrixFromEdges(numPeople, numPeople, follows)

	printGroups([]string{"ava", "bo", "cy", "dee", "eli", "flo", "kim"}, clustering.PeerPressure[float64](context.Background(), g))
	// Output:
	// 0: ava bo cy kim
	// 1: dee eli flo
}

// ExampleSpectral splits a road network into three regions. Towns within
// a region are linked by local roads; a single highway joins each pair
// of regions.
func ExampleSpectral() {
	names := []string{
		"ashby", "brill", "corfe", "dent", // north
		"elton", "frome", "goole", "hythe", // coast
		"islip", "jarrow", "kendal", "leek", // hills
	}

	var roads []graphblas.Edge[float64]
	for region := 0; region < 3; region++ {
		for i := 0; i < 4; i++ {
			for j := i + 1; j < 4; j++ {
				if (i+j)%3 != 0 { // not every pair of towns has a road
					roads = append(roads, graphblas.Edge[float64]{From: region*4 + i, To: region*4 + j, Weight: 1})
				}
			}
		}
	}
	roads = append(roads,
		graphblas.Edge[float64]{From: 3, To: 4, Weight: 1},  // dent - elton
		graphblas.Edge[float64]{From: 7, To: 8, Weight: 1},  // hythe - islip
		graphblas.Edge[float64]{From: 11, To: 0, Weight: 1}, // leek - ashby
	)

	g := graphblas.NewCSRMatrixFromEdges(len(names), len(names), roads)

	printGroups(names, clustering.Spectral[float64](context.Background(), g, 3))
	// Output:
	// 0: ashby brill corfe dent
	// 1: elton frome goole hythe
	// 2: islip jarrow kendal leek
}

// ExampleLocal finds the circle around Gus alone, rather than partitioning
// everyone: probability spreads out from Gus and the sweep stops at the
// climbing gym, where the fewest friendships leave the group.
func ExampleLocal() {
	names, edges := friendships()
	g := graphblas.NewCSRMatrixFromEdges(len(names), len(names), edges)

	for _, v := range clustering.Local[float64](context.Background(), g, 6) {
		fmt.Println(names[v])
	}
	// Output:
	// eve
	// fay
	// gus
	// hal
}
