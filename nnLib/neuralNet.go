package nn

// ------------------------------------------------------------------------------------------------
// Neurons
// ------------------------------------------------------------------------------------------------

type Neuron struct {
	weights []*Value
	bias    *Value
}

func NewNeuron(totalInputs int) *Neuron {
	neuron := &Neuron{
		weights: make([]*Value, totalInputs),
		bias:    NewValue(0.0),
	}

	for i := range totalInputs {
		neuron.weights[i] = NewValue(RandFloatRange(-1.0, 1.0))
	}

	return neuron
}

func NeuronForwardProp[T ValueOpTypes](neuron *Neuron, values []T, nonLin bool) *Value {
	sum := NewValue(0.0)

	for i, weight := range neuron.weights {
		sum = Add(
			sum,
			Mul(values[i], weight),
		)
	}

	sum = Add(sum, neuron.bias)

	if nonLin {
		return Relu(sum)
	}

	return sum
}

func GetNeuronParams(neuron *Neuron) []*Value {
	return append(neuron.weights, neuron.bias)
}

// ------------------------------------------------------------------------------------------------
// Layer
// ------------------------------------------------------------------------------------------------

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(totalInputs, totalNeurons int) *Layer {
	layer := &Layer{Neurons: make([]*Neuron, totalNeurons)}

	for i := range totalNeurons {
		layer.Neurons[i] = NewNeuron(totalInputs)
	}

	return layer
}

func LayerForwardProp[T ValueOpTypes](layer *Layer, values []T, nonLin bool) []*Value {
	newValues := []*Value{}

	for _, neuron := range layer.Neurons {
		newValues = append(newValues, NeuronForwardProp(neuron, values, nonLin))
	}

	return newValues
}

func GetLayerParams(layer *Layer) []*Value {
	layerParams := []*Value{}

	for _, neuron := range layer.Neurons {
		layerParams = append(layerParams, GetNeuronParams(neuron)...)
	}

	return layerParams
}

// ------------------------------------------------------------------------------------------------
// Complete Neural Net
// ------------------------------------------------------------------------------------------------

type MultiLayerPerceptron struct {
	Layers  []*Layer
	NonLins []bool
}

func NewMLP(totalInputs int, layerSizes []int) *MultiLayerPerceptron {
	mlp := &MultiLayerPerceptron{
		Layers:  make([]*Layer, len(layerSizes)),
		NonLins: make([]bool, len(layerSizes)),
	}

	sizes := make([]int, len(layerSizes)+1)
	sizes[0] = totalInputs

	for i := 1; i < len(sizes); i++ {
		sizes[i] = layerSizes[i-1]
	}

	for i := range layerSizes {
		mlp.Layers[i] = NewLayer(sizes[i], sizes[i+1])
		mlp.NonLins[i] = i != len(layerSizes)-1
	}

	return mlp
}

func ForwardProp[T ValueOpTypes](mlp *MultiLayerPerceptron, values []T) []*Value {
	var result []*Value
	for i, layer := range mlp.Layers {
		if i == 0 {
			result = LayerForwardProp(layer, values, mlp.NonLins[i])
		} else {
			result = LayerForwardProp(layer, result, mlp.NonLins[i])
		}
	}

	return result
}

func GetMLPParams(mlp *MultiLayerPerceptron) []*Value {
	mlpParams := []*Value{}

	for _, layer := range mlp.Layers {
		mlpParams = append(mlpParams, GetLayerParams(layer)...)
	}

	return mlpParams
}

// ------------------------------------------------------------------------------------------------
