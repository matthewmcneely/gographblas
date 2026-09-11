// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package centrality_test

import (
	"context"
	"fmt"

	"github.com/rossmerr/graphblas"
	"github.com/rossmerr/graphblas/centrality"
)

// ExamplePageRank ranks the pages of a small website. Every page links
// back to the home page, so it collects the most rank; the contact page
// has no in-links and keeps only its teleport share.
func ExamplePageRank() {
	const (
		home = iota
		about
		blog
		contact
		numPages
	)

	links := graphblas.NewCSRMatrixFromEdges(numPages, numPages, []graphblas.Edge[float64]{
		{From: home, To: about, Weight: 1},
		{From: home, To: blog, Weight: 1},
		{From: about, To: home, Weight: 1},
		{From: blog, To: home, Weight: 1},
		{From: blog, To: about, Weight: 1},
		{From: contact, To: home, Weight: 1},
	})

	ranks := centrality.PageRank[float64](context.Background(), links)

	for i, name := range []string{"home", "about", "blog", "contact"} {
		fmt.Printf("%s: %.3f\n", name, ranks.AtVec(i))
	}
	// Output:
	// home: 0.429
	// about: 0.313
	// blog: 0.220
	// contact: 0.038
}

// ExampleBetweenness finds the broker in a small org: every path between
// the two analysts and the two field teams runs through the lead, so the
// lead carries all four pairs.
func ExampleBetweenness() {
	const (
		ana = iota
		ben
		lead
		cam
		dee
		numPeople
	)

	reports := graphblas.NewCSRMatrixFromEdges(numPeople, numPeople, []graphblas.Edge[float64]{
		{From: ana, To: lead, Weight: 1},
		{From: ben, To: lead, Weight: 1},
		{From: lead, To: cam, Weight: 1},
		{From: lead, To: dee, Weight: 1},
	})

	bc := centrality.Betweenness[float64](context.Background(), reports)

	for i, name := range []string{"ana", "ben", "lead", "cam", "dee"} {
		fmt.Printf("%s: %v\n", name, bc.AtVec(i))
	}
	// Output:
	// ana: 0
	// ben: 0
	// lead: 4
	// cam: 0
	// dee: 0
}

// ExampleCloseness scores how quickly each depot in a delivery network
// reaches the rest, on weighted travel times. Dispatch reaches everything
// cheaply; the final depot reaches nothing.
func ExampleCloseness() {
	const (
		dispatch = iota
		north
		south
		depot
		numSites
	)

	routes := graphblas.NewCSRMatrixFromEdges(numSites, numSites, []graphblas.Edge[float64]{
		{From: dispatch, To: north, Weight: 1},
		{From: dispatch, To: south, Weight: 2},
		{From: north, To: depot, Weight: 3},
		{From: south, To: depot, Weight: 1},
	})

	c := centrality.Closeness[float64](context.Background(), routes)

	for i, name := range []string{"dispatch", "north", "south", "depot"} {
		fmt.Printf("%s: %.3f\n", name, c.AtVec(i))
	}
	// Output:
	// dispatch: 0.500
	// north: 0.111
	// south: 0.333
	// depot: 0.000
}
