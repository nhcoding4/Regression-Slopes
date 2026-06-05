package nnLib

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// ------------------------------------------------------------------------------------------------
// Types
// ------------------------------------------------------------------------------------------------

type Operation byte

const (
	None Operation = iota
	Add
	Exp
	Mul
	Pow
	ReLU
	TanH
)

func (o Operation) String() string {
	switch o {
	case None:
		return "None"
	case Add:
		return "Add"
	case Exp:
		return "Exp"
	case Mul:
		return "Mul"
	case Pow:
		return "Pow"
	case ReLU:
		return "ReLU"
	case TanH:
		return "TanH"
	default:
		return "Unknown OpType"
	}
}

type Value struct {
	Data      float64
	Gradient  float64
	OpType    Operation
	idx0      int
	idx1      int
	constNode bool
}

type Graph struct {
	Values    []Value
	powers    map[int]float64
	order     []int
	weightEnd int
	biasEnd   int
}

func NewGraph() Graph {
	return Graph{
		powers: make(map[int]float64),
	}
}

func (g *Graph) NewValue(data float64, opType Operation, parents ...int) int {
	graphIdx := len(g.Values)

	value := Value{
		Data:   data,
		OpType: opType,
	}

	switch len(parents) {
	case 0:
		value.constNode = true
	case 1:
		value.idx0 = parents[0]
	case 2:
		value.idx0 = parents[0]
		value.idx1 = parents[1]
	default:
		panic("Error: NewValue() was passed an invalid amount of parent indices. A new value can only ever have 2 parents.")
	}

	g.Values = append(g.Values, value)

	return graphIdx
}

// ------------------------------------------------------------------------------------------------
// Utility
// ------------------------------------------------------------------------------------------------

func (v *Value) String() string {
	return fmt.Sprintf("Value(data: %0.3f, grad: %0.3f, op: %s)", v.Data, v.Gradient, v.OpType.String())
}

func (g *Graph) GetValue(idx int) *Value {
	return &g.Values[idx]
}

func (g *Graph) GetData(idx int) float64 {
	return g.Values[idx].Data
}

func (g *Graph) GradDescent(learningRate float64) {
	for i := range g.Values {
		val := g.GetValue(i)
		if val.constNode || val.OpType != None {
			continue
		}

		val.Data -= learningRate * val.Gradient
	}
}

func (g *Graph) PostTrainingCleanup() {
	g.Values = g.Values[:g.biasEnd]
	g.powers = make(map[int]float64)
}

func randomFloatRange(start, stop float64) float64 {
	return start + rand.Float64()*(stop-start)
}

// ------------------------------------------------------------------------------------------------
// Operations
// ------------------------------------------------------------------------------------------------

func (g *Graph) Add(a, b int) int {
	return g.NewValue(0.0, Add, a, b)
}

func (g *Graph) Exp(a int) int {
	return g.NewValue(0.0, Exp, a)
}

func (g *Graph) Mul(a, b int) int {
	return g.NewValue(0.0, Mul, a, b)
}

func (g *Graph) Pow(a int, power float64) int {
	pow := g.NewValue(0.0, Pow, a)
	g.powers[pow] = power

	return pow
}

func (g *Graph) Div(a int, b int) int {
	return g.Mul(a, g.Pow(b, -1.0))
}

func (g *Graph) Negate(a int) int {
	return g.Mul(a, g.NewValue(-1.0, None))
}

func (g *Graph) Sub(a, b int) int {
	return g.Add(a, g.Negate(b))
}

// ------------------------------------------------------------------------------------------------
// Transforms
// ------------------------------------------------------------------------------------------------

func (g *Graph) MeanSqrErr(want []float64, got []int) int {
	sum := g.NewValue(0.0, None)

	for i, val := range got {
		difference := g.Pow(g.Sub(g.NewValue(want[i], None), val), 2.0)
		sum = g.Add(sum, difference)
	}

	return g.Div(sum, g.NewValue(float64(len(got)), None))
}

func (g *Graph) ReLU(a int) int {
	return g.NewValue(0.0, ReLU, a)
}

func (g *Graph) TanH(a int) int {
	return g.NewValue(0.0, TanH, a)
}

// ------------------------------------------------------------------------------------------------
// Create Graph
// ------------------------------------------------------------------------------------------------

func (g *Graph) makePersistentValues(inputs int, shape []int) {
	layers := len(shape)
	sizes := make([]int, layers+1)
	sizes[0] = inputs

	for i, layerSize := range shape {
		sizes[i+1] = layerSize
	}

	for i := range layers {
		curInputs := sizes[i]
		neurons := sizes[i+1]

		for range neurons {
			for range curInputs {
				newWeight := g.NewValue(randomFloatRange(-1.0, 1.0), None)
				g.GetValue(newWeight).constNode = false
			}
		}
	}

	g.weightEnd = len(g.Values)

	for i := range layers {
		neurons := sizes[i+1]

		for range neurons {
			newBias := g.NewValue(0.0, None)
			g.GetValue(newBias).constNode = false
		}
	}

	g.biasEnd = len(g.Values)
}

func (g *Graph) CreateGraph(inputs []float64, shape []int, remake bool) []int {
	startingInputs := len(inputs)

	if remake || len(g.Values) == 0 {
		g.makePersistentValues(startingInputs, shape)
	}

	curInputs := make([]int, startingInputs)

	for i, input := range inputs {
		curInputs[i] = g.NewValue(input, None)
	}

	weightIdx := 0
	biasIdx := g.weightEnd

	for i, layerSize := range shape {
		newInputs := []int{}

		for range layerSize {
			sum := g.NewValue(0.0, None)

			for _, input := range curInputs {
				weightVal := g.Mul(input, weightIdx)
				weightIdx++
				sum = g.Add(sum, weightVal)
			}

			sum = g.Add(sum, biasIdx)
			biasIdx++

			if i < len(shape)-1 {
				newInputs = append(newInputs, g.ReLU(sum))
			} else {
				newInputs = append(newInputs, sum)
			}
		}

		curInputs = newInputs
	}

	return curInputs
}

// ------------------------------------------------------------------------------------------------
// Forward / Backward Propagation
// ------------------------------------------------------------------------------------------------

func (g *Graph) buildTopOrder(root int) {
	g.order = []int{}

	var iter func(int, map[int]byte)

	iter = func(cur int, visited map[int]byte) {
		if _, ok := visited[cur]; !ok {
			visited[cur] = 1
			value := g.GetValue(cur)

			switch value.OpType {
			case Add, Mul:
				iter(value.idx0, visited)
				iter(value.idx1, visited)

			case Exp, Pow, ReLU, TanH:
				iter(value.idx0, visited)
			}

			g.order = append(g.order, cur)
		}
	}

	iter(root, make(map[int]byte))
}

func (g *Graph) BackProp(root int, rebuild bool) {
	if rebuild || len(g.order) == 0 {
		g.buildTopOrder(root)
	}

	for i := range g.Values {
		g.GetValue(i).Gradient = 0
	}

	g.Values[root].Gradient = 1

	for i := len(g.order) - 1; i >= 0; i-- {
		curRoot := g.GetValue(g.order[i])

		switch curRoot.OpType {
		case Add:
			g.Values[curRoot.idx0].Gradient += curRoot.Gradient
			g.Values[curRoot.idx1].Gradient += curRoot.Gradient

		case Exp:
			g.Values[curRoot.idx0].Gradient += curRoot.Data * curRoot.Gradient

		case Mul:
			x := g.GetValue(curRoot.idx0)
			y := g.GetValue(curRoot.idx1)
			x.Gradient += y.Data * curRoot.Gradient
			y.Gradient += x.Data * curRoot.Gradient

		case Pow:
			x := g.GetValue(curRoot.idx0)
			power := g.powers[g.order[i]]
			x.Gradient += (power * math.Pow(x.Data, power-1)) * curRoot.Gradient

		case ReLU:
			var base float64
			if curRoot.Data > 0 {
				base = 1.0
			}

			g.Values[curRoot.idx0].Gradient += base * curRoot.Gradient

		case TanH:
			tanH := curRoot.Data
			g.Values[curRoot.idx0].Gradient += (1 - tanH*tanH) * curRoot.Gradient
		}
	}
}

func (g *Graph) ForwardProp(root int, rebuild bool) {
	if len(g.order) == 0 || rebuild {
		g.buildTopOrder(root)
	}

	for i := 0; i < len(g.order); i++ {
		curRoot := g.GetValue(g.order[i])

		switch curRoot.OpType {
		case Add:
			curRoot.Data = g.GetData(curRoot.idx0) + g.GetData(curRoot.idx1)

		case Exp:
			curRoot.Data = math.Exp(g.GetData(curRoot.idx0))

		case Mul:
			curRoot.Data = g.GetData(curRoot.idx0) * g.GetData(curRoot.idx1)

		case Pow:
			curRoot.Data = math.Pow(g.GetData(curRoot.idx0), g.powers[g.order[i]])

		case ReLU:
			data := g.GetData(curRoot.idx0)
			if data < 0 {
				data = 0
			}

			curRoot.Data = data

		case TanH:
			data := g.GetData(curRoot.idx0)
			exp := math.Exp(2 * data)
			tanH := (exp - 1) / (exp + 1)
			curRoot.Data = tanH
		}
	}
}

// ------------------------------------------------------------------------------------------------
