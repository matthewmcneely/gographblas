// Copyright (c) 2018 Ross Merrigan
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

package binaryop

import (
	"math"

	"github.com/rossmerr/graphblas/constraints"
)

// Semiring pairs an additive monoid with a multiplicative binary operator.
// A matrix multiply over a semiring reduces with the monoid, seeded at its
// Zero, and combines matrix and vector elements with the multiplicative
// operator. Swapping the semiring turns the same multiply into different
// graph algorithms: (+, *) is ordinary arithmetic, (min, +) advances
// shortest-path distances, (max, min) computes widest-path capacities.
type Semiring[T constraints.None] interface {
	Add() MonoID[T]
	Multiply() BinaryOp[T]
}

type semiring[T constraints.None] struct {
	add      MonoID[T]
	multiply BinaryOp[T]
}

func (s *semiring[T]) Add() MonoID[T] {
	return s.add
}

func (s *semiring[T]) Multiply() BinaryOp[T] {
	return s.multiply
}

// NewSemiring returns a Semiring over the given additive monoid and
// multiplicative operator.
func NewSemiring[T constraints.None](add MonoID[T], multiply BinaryOp[T]) Semiring[T] {
	return &semiring[T]{add: add, multiply: multiply}
}

// PlusTimes is the conventional arithmetic semiring: reduce with addition
// (identity 0), combine with multiplication.
func PlusTimes[T constraints.Number]() Semiring[T] {
	return NewSemiring[T](NewMonoID(0, Addition[T]()), Multiplication[T]())
}

// MinPlus is the tropical semiring: reduce with minimum (identity +Inf),
// combine with addition. One matrix-vector multiply over MinPlus advances
// shortest-path distances by a single edge relaxation step.
func MinPlus[T constraints.Float]() Semiring[T] {
	return NewSemiring[T](NewMonoID(T(math.Inf(1)), Minimum[T]()), Addition[T]())
}

// MaxMin is the bottleneck semiring: reduce with maximum (identity -Inf),
// combine with minimum. One matrix-vector multiply over MaxMin advances
// widest-path (maximum bottleneck) capacities by one step.
func MaxMin[T constraints.Float]() Semiring[T] {
	return NewSemiring[T](NewMonoID(T(math.Inf(-1)), Maximum[T]()), Minimum[T]())
}
