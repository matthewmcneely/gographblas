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
