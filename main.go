package main

import (
	"fmt"
	"image/color"
	"math"
	nn "nnLib"
	"os"
	"sync"
	"time"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/font"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func createData(slope float64, size int) ([]float64, []float64) {
	xValues := nn.RandNormArray(size)
	yValues := nn.RandNormArray(size)

	for i, val := range yValues {
		yValues[i] = slope*xValues[i] + val/2.0
	}

	return xValues, yValues
}

func trainModel(x, y []float64) ([]float64, []float64) {
	const lr = 0.05
	const epochs = 500
	expected := y

	g := nn.NewGraph()
	results := make([]int, len(x)) // Here
	losses := make([]float64, epochs)

	for i := range epochs {
		for j, val := range x {
			results[j] = g.CreateGraph([]float64{val}, []int{4, 4, 1}, false)[0]
		}

		mse := g.MeanSqrErr(expected, results)
		g.ForwardProp(mse, false)
		losses[i] = g.GetData(mse)

		g.BackProp(mse, false)
		g.GradDescent(lr)

		if i != epochs-1 {
			g.PostTrainingCleanup()
		}
	}

	out := make([]float64, len(x))
	for i := range out {
		out[i] = g.GetData(results[i])
	}

	return out, losses
}

func getColumn(data [][][]float64, column int) []float64 {
	results := []float64{}

	for _, row := range data {
		for _, col := range row {
			results = append(results, col[column])
		}
	}

	return results
}

func getExperimentData(data [][]float64, column int) []float64 {
	results := []float64{}

	for _, exp := range data {
		results = append(results, exp[column])
	}

	return results
}

func mean(data []float64) float64 {
	sum := 0.0

	for _, x := range data {
		sum += x
	}

	return sum / float64(len(data))
}

func runExperiments(experiments int) {
	start := time.Now()

	slopes := nn.Linspace(-2, 2, 21, true)
	results := make([][][]float64, len(slopes))

	for i := range results {
		results[i] = make([][]float64, experiments)
		for j := range experiments {
			results[i][j] = make([]float64, 2)
		}
	}

	var wg sync.WaitGroup

	for i := range len(slopes) {
		wg.Go(func() {
			for j := range experiments {
				x, y := createData(slopes[i], 50)
				yhat, losses := trainModel(x, y)
				results[i][j][0] = losses[len(losses)-1]
				results[i][j][1] = stat.Correlation(y, yhat, nil)
			}
		})
	}
	wg.Wait()

	for y, row := range results {
		for x, col := range row {
			for z, num := range col {
				if math.IsNaN(num) {
					results[y][x][z] = 0
				}
			}
		}
	}

	end := time.Since(start)
	fmt.Printf("Completed %d experiments in %v\n", experiments, end)

	lossMeans := plotter.XYs{}
	corrcoefMeans := plotter.XYs{}

	for i, experiment := range results {
		x := slopes[i]
		lossMeans = append(lossMeans, plotter.XY{X: x, Y: mean(getExperimentData(experiment, 0))})
		corrcoefMeans = append(corrcoefMeans, plotter.XY{X: x, Y: mean(getExperimentData(experiment, 1))})
	}

	titleSize := font.Length(20.0)
	glyphRadius := vg.Length(4.0)

	plots := make([][]*plot.Plot, 1)
	subplots := []*plot.Plot{}

	// Plot loss average
	lossPlot := plot.New()
	subplots = append(subplots, lossPlot)
	lossPlot.Title.Text = "Loss"
	lossPlot.Title.TextStyle.Font.Size = titleSize
	lossPlot.X.Label.Text = "Slope"

	line, scatter, err := plotter.NewLinePoints(lossMeans)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Radius = glyphRadius

	lossPlot.Add(line, scatter)

	// Plot Model Performance
	perfPlot := plot.New()
	subplots = append(subplots, perfPlot)
	perfPlot.Title.Text = "Model Performance"
	perfPlot.Title.TextStyle.Font.Size = titleSize
	perfPlot.X.Label.Text = "Slope"
	perfPlot.Y.Label.Text = "Real-Predicted Correlation"

	line, scatter, err = plotter.NewLinePoints(corrcoefMeans)
	if err != nil {
		panic(err)
	}
	magenta := color.RGBA{255, 0, 255, 255}
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Radius = glyphRadius
	scatter.Color = magenta
	line.Color = magenta

	perfPlot.Add(line, scatter)

	plots[0] = subplots

	img := vgimg.New(vg.Inch*20, vg.Inch*10)
	drawCanvas := draw.New(img)
	tiles := draw.Tiles{Rows: 1, Cols: 2}
	canvas := plot.Align(plots, tiles, drawCanvas)

	for y, row := range canvas {
		for x := range row {
			plots[y][x].Draw(canvas[y][x])
		}
	}

	writer, err := os.Create("results.png")
	if err != nil {
		panic(err)
	}

	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(writer); err != nil {
		panic(err)
	}
}

func main() {
	runExperiments(50)
}
