package graphblas

import (
	"github.com/matthewmcneely/gographblas/binaryop"
	"github.com/matthewmcneely/gographblas/constraints"
)

func Zero[T any]() T {
	return *new(T)
}

func IsZero[T comparable](v T) bool {
	return v == *new(T)
}

func Default[T any]() T {
	return *new(T)
}

func DefaultMonoIDAddition[T constraints.Number]() binaryop.MonoID[T] {
	d := Default[T]()
	add := binaryop.Addition[T]()
	return binaryop.NewMonoID(d, add)
}

func DefaultMonoIDMaximum[T constraints.Number]() binaryop.MonoID[T] {
	d := Default[T]()
	max := binaryop.Maximum[T]()
	return binaryop.NewMonoID(d, max)
}
