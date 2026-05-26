package nn

import (
	"fmt"
	"math"
)

// --------------------------------------------------------------------------------------------------------------------
// Types
// --------------------------------------------------------------------------------------------------------------------

type Value struct {
	Data     float64
	Gradient float64
	Backward func()
	Children map[*Value]byte
}

type ValueOpTypes interface {
	float64 | *Value
}

func NewValue(data float64) *Value {
	return &Value{Data: data, Children: make(map[*Value]byte), Backward: func() {}}
}

func newValueWithChildren(data float64, children []*Value) *Value {
	newValue := NewValue(data)

	for _, child := range children {
		newValue.Children[child] = 1
	}

	return newValue
}

// --------------------------------------------------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------------------------------------------------

func convertToValue[T ValueOpTypes](callee string, b T) *Value {
	var other *Value = nil

	switch y := any(b).(type) {
	case float64:
		other = NewValue(y)
	case *Value:
		other = y
	default:
		panic(
			fmt.Sprintf(
				"Fatal error: %s() was passed an invalid type (%v). Only floats or values can be used.\n",
				callee, b,
			),
		)
	}

	return other
}

// --------------------------------------------------------------------------------------------------------------------
// Utility
// --------------------------------------------------------------------------------------------------------------------

func ToString(a *Value) string {
	return fmt.Sprintf("Value{Data: %0.3f, Gradient: %0.3f}", a.Data, a.Gradient)
}

// --------------------------------------------------------------------------------------------------------------------
// Operations
// --------------------------------------------------------------------------------------------------------------------

func Add[T ValueOpTypes, U ValueOpTypes](a T, b U) *Value {
	const callee = "ValueAdd"
	x := convertToValue(callee, a)
	y := convertToValue(callee, b)

	out := newValueWithChildren(x.Data+y.Data, []*Value{x, y})
	out.Backward = func() {
		x.Gradient += out.Gradient
		y.Gradient += out.Gradient
	}

	return out
}

func Mul[T ValueOpTypes, U ValueOpTypes](a T, b U) *Value {
	const callee = "ValueMul"
	x := convertToValue(callee, a)
	y := convertToValue(callee, b)

	out := newValueWithChildren(x.Data*y.Data, []*Value{x, y})
	out.Backward = func() {
		x.Gradient += y.Data * out.Gradient
		y.Gradient += x.Data * out.Gradient
	}

	return out
}

func Pow[T ValueOpTypes](a T, pow float64) *Value {
	x := convertToValue("ValuePow", a)

	out := newValueWithChildren(math.Pow(x.Data, pow), []*Value{x})
	out.Backward = func() {
		x.Gradient += (pow * math.Pow(x.Data, pow-1)) * out.Gradient
	}

	return out
}

func Exp(a *Value) *Value {
	out := newValueWithChildren(math.Exp(a.Data), []*Value{a})
	out.Backward = func() {
		a.Gradient += out.Data * out.Gradient
	}

	return out
}

func Negate(a *Value) *Value {
	return Mul(a, -1.0)
}

func Sub[T ValueOpTypes, U ValueOpTypes](a T, b U) *Value {
	const name = "Sub"
	x := convertToValue(name, a)
	y := convertToValue(name, b)
	return Add(x, Negate(y))
}

func Div[T ValueOpTypes, U ValueOpTypes](a T, b U) *Value {
	const name = "Div"
	x := convertToValue(name, a)
	y := convertToValue(name, b)

	if y.Data == 0 {
		panic(fmt.Sprintf("%s() - Attempted a by zero\n", name))
	}

	return Mul(x, Pow(y, -1.0))
}

// --------------------------------------------------------------------------------------------------------------------
// Transformation Functions
// --------------------------------------------------------------------------------------------------------------------

func MeanSquaredError(want []float64, got []*Value) *Value {
	sum := NewValue(0.0)

	for i, val := range got {
		err := Pow(Sub(want[i], val), 2)
		sum = Add(sum, err)
	}

	return Div(sum, float64(len(got)))
}

func Relu(a *Value) *Value {
	var data float64
	if a.Data > 0 {
		data = a.Data
	}

	out := newValueWithChildren(data, []*Value{a})
	out.Backward = func() {
		var base float64
		if out.Data > 0 {
			base = 1.0
		}

		a.Gradient += base * out.Gradient
	}

	return out
}

func TanH(a *Value) *Value {
	x := a.Data
	tanH := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)

	out := newValueWithChildren(tanH, []*Value{a})
	out.Backward = func() {
		a.Gradient += (1 - tanH*tanH) * out.Gradient
	}

	return out
}

// --------------------------------------------------------------------------------------------------------------------
// Back Propagation
// --------------------------------------------------------------------------------------------------------------------

func buildTopologicalOrder(root *Value, topOrder []*Value, visited map[*Value]byte) []*Value {
	if _, ok := visited[root]; !ok {
		visited[root] = 1

		for key := range root.Children {
			topOrder = buildTopologicalOrder(key, topOrder, visited)
		}

		topOrder = append(topOrder, root)
	}

	return topOrder
}

func ValueBackwardPropagate(root *Value) {
	visited := make(map[*Value]byte)
	topologicalOrder := buildTopologicalOrder(root, []*Value{}, visited)

	root.Gradient = 1
	for i := len(topologicalOrder) - 1; i >= 0; i-- {
		topologicalOrder[i].Backward()
	}
}

// --------------------------------------------------------------------------------------------------------------------
