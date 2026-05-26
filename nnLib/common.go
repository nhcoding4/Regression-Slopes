package nn

import (
	"math"
	"math/rand/v2"
)

// ------------------------------------------------------------------------------------------------
// Unassociated Functions
// ------------------------------------------------------------------------------------------------

func Linspace(start, stop, elements int, endpoint bool) []float64 {
	if endpoint {
		elements--
	}

	data := []float64{}
	span := float64(stop - start)
	step := 1.0 / float64(elements)

	if span <= 0 {
		return data
	}

	if endpoint {
		elements++
	}

	for i := range elements {
		point := float64(start) + float64(i)*span*step
		data = append(data, point)
	}

	return data
}

func RandFloatRange(start, stop float64) float64 {
	return start + rand.Float64()*(stop-start)
}

func RandNorm() float64 {
	a := rand.Float64()
	b := rand.Float64()

	if a == 0 {
		b = 1e-10
	}

	res := math.Sqrt(-2.0 * math.Log(a) * math.Cos(2.0*math.Pi*b))
	if math.IsNaN(res) {
		return 0.0
	}

	return res
}

func RandNormArray(size int) []float64 {
	out := make([]float64, size)

	for i := range size {
		out[i] = RandNorm()
	}

	return out
}

// ------------------------------------------------------------------------------------------------
