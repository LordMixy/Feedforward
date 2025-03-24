#include <iostream>
#include <array>
#include <cmath>
#include <map>
#include <ranges>
#include <vector>
#include <ctime>
#include <cstdio>
#include <random>

namespace NeuralNetwork
{
	template <std::size_t N> 
	using vec = std::array<double, N>;

	std::mt19937 randomEngine;
	std::uniform_real_distribution<double> xavierDistribution;
	
	double sigmoid(const double& x)
	{
		return 0.5 * (1.0 + std::tanh(0.5 * x));
	}

	double sigmoidDerivative(const double& x)
	{
		return x * (1.0 - x);
	}
	
	template <std::size_t N, std::size_t M> 
	struct Perceptrons
	{	
		std::array<vec<N>, M> weights;
		vec<M> bias;

		void xavierWeightInitialize(const std::size_t& output)
		{
			double distributionLimit = std::sqrt(6.0 / (N + output));
			xavierDistribution.param(std::uniform_real_distribution<double>::param_type(-distributionLimit, distributionLimit));

			for (std::size_t i = 0; i < M; ++i)
			{
				std::generate(weights[i].begin(), weights[i].end(), [&] () { 
					return xavierDistribution(randomEngine); 
				});
				bias[i] = 0;
			}
		}
		
		void updateParameters(const vec<N>& inputs, const vec<M>& errorDelta, const double& learningRate)
		{
			for (std::size_t i = 0; i < M; ++i)
			{
				double delta = learningRate * errorDelta[i];
				for (std::size_t j = 0; j < N; ++j)
				{
					weights[i][j] += inputs[j] * delta;
				}
				bias[i] += delta;
			}
		}

		vec<M> feedForward(const vec<N>& inputs)
		{
            vec<M> sigmoids;
            for (std::size_t i = 0; i < M; ++i)
            {
                double dotProduct = bias[i];
                for (std::size_t j = 0; j < N; ++j)
                {
                    dotProduct += weights[i][j] * inputs[j];
                }

                sigmoids[i] = sigmoid(dotProduct);
            }
            return sigmoids;
		}
	};

	template <std::size_t N, std::size_t M, std::size_t K> 
	struct HiddenLayers
	{
		Perceptrons<N, M> firstLayer;
		std::array<Perceptrons<M, M>, K - 1> middleLayers;
		double learningRate;
		
		vec<M> computeLayerOutput(const vec<N>& inputs)
		{
			vec<M> layerOutput = firstLayer.feedForward(inputs);
			return layerOutput;
		}

		vec<M> computeLayerOutput(const std::size_t& layer, const vec<M>& inputs)
		{
			vec<M> layerOutput = middleLayers[layer].feedForward(inputs);
			return layerOutput;
		}

		vec<M> computeErrorsDelta(const vec<M>& layerOutput, const vec<M>& weights, const double& errorDelta)
		{
			vec<M> errorsDelta;
			for (std::size_t i = 0; i < M; ++i)
			{
				double layerError = errorDelta * weights[i];
				errorsDelta[i] = layerError * sigmoidDerivative(layerOutput[i]);
			}
			return errorsDelta;
		}
		
		void backPropagate(const vec<N>& input, const vec<M>& weights, const double& errorDelta)
		{
			vec<M> layerOutput = computeLayerOutput(input);
			vec<M> errorsDelta = computeErrorsDelta(layerOutput, weights, errorDelta);
			
			this->firstLayer.updateParameters(input, errorsDelta, learningRate);

			vec<M> nextLayerOutput;
			for (std::size_t i = 0; i < K - 1; ++i, layerOutput = nextLayerOutput)
			{
				nextLayerOutput = computeLayerOutput(i, layerOutput);
				errorsDelta = computeErrorsDelta(nextLayerOutput, weights, errorDelta);
				
				this->middleLayers[i].updateParameters(layerOutput, errorsDelta, learningRate);	
			}
		}
	};
	
	template <std::size_t N, std::size_t M, std::size_t K> 
	struct FeedForwardNetwork 
	{
		HiddenLayers<N, M, K> hiddenLayers;
		Perceptrons<M, 1> output;

		std::vector<vec<N>> inputs;
		std::vector<double> targets;
				
		FeedForwardNetwork(const std::map<vec<N>, double>& trainData, const double& learningRate)
		{	
			hiddenLayers.learningRate = learningRate;

			hiddenLayers.firstLayer.xavierWeightInitialize(M);
			for (auto& layer : hiddenLayers.middleLayers)
			{
				layer.xavierWeightInitialize(M);
			}			
			output.xavierWeightInitialize(1);
			
            for (const auto& [key, value] : trainData) 
            {
            	inputs.push_back(key);
            	targets.push_back(value);
            }
		}

		vec<M> computeHiddenLayersOutput(const vec<N>& inputs)
		{
			vec<M> hiddenLayerOutput = hiddenLayers.computeLayerOutput(inputs);
			for (std::size_t i = 0; i < K - 1; ++i)
			{
				hiddenLayerOutput = hiddenLayers.computeLayerOutput(i, hiddenLayerOutput);
			}
			return hiddenLayerOutput;
		}
		
		double feedForward(const vec<N>& inputs)
		{
			// banalmente non e' altro che f^1(f^2(...f^N(input)));
			// in questo caso e' nella forma: outputPerceptron.feedForward(hiddenLayer1(...hiddenLayerN(input)))
			vec<M> hiddenLayerOutput = computeHiddenLayersOutput(inputs);
			return output.feedForward(hiddenLayerOutput)[0];
		}
		
		void train(const std::size_t& iterations)
		{
			const auto iSize = inputs.size();
			for (std::size_t k = 0; k < iterations; ++k)
			{	
				for (std::size_t i = 0; i < iSize; ++i)
				{
					auto currentInput = inputs[i];
					vec<M> hiddenLayerOutput = computeHiddenLayersOutput(currentInput);				

					double output = this->output.feedForward(hiddenLayerOutput)[0];
					double errorDelta = (targets[i] - output) * sigmoidDerivative(output);

					this->output.updateParameters(hiddenLayerOutput, { errorDelta }, hiddenLayers.learningRate);

					hiddenLayers.backPropagate(currentInput, this->output.weights[0], errorDelta);
				}
			}
		}
	};
}

#define XOR_LAYERS 2
#define XOR_LAYER_PERCEPTRONS 3

#define XNOR_LAYERS 4
#define XNOR_LAYER_PERCEPTRONS 5

#define TRAIN_ITERATIONS 100000
#define LEARNING_RATE 0.1

int main()
{
	using namespace NeuralNetwork;
	using namespace std::views;

	const std::map<vec<3>, double> trainDataXOR =
	{
		{ { 0, 0, 0 }, 0 },	
		{ { 0, 0, 1 }, 1 },	
		{ { 0, 1, 0 }, 1 },	
		{ { 0, 1, 1 }, 1 },	
		{ { 1, 0, 0 }, 1 },	
		{ { 1, 0, 1 }, 1 },	
		{ { 1, 1, 0 }, 1 },	
		{ { 1, 1, 1 }, 0 },	
	};

	const std::map<vec<3>, double> trainDataXNOR =
	{
		{ { 0, 0, 0 }, 1 },	
		{ { 0, 0, 1 }, 0 },	
		{ { 0, 1, 0 }, 0 },	
		{ { 0, 1, 1 }, 1 },	
		{ { 1, 0, 0 }, 0 },	
		{ { 1, 0, 1 }, 1 },	
		{ { 1, 1, 0 }, 1 },	
		{ { 1, 1, 1 }, 0 },	
	};
		
	FeedForwardNetwork<3, XOR_LAYER_PERCEPTRONS, XOR_LAYERS> xorNetwork { trainDataXOR, LEARNING_RATE };
	FeedForwardNetwork<3, XNOR_LAYER_PERCEPTRONS, XNOR_LAYERS> xnorNetwork { trainDataXNOR, LEARNING_RATE };

	xorNetwork.train(TRAIN_ITERATIONS);
	xnorNetwork.train(TRAIN_ITERATIONS);
	
	for (const auto& input : keys(trainDataXOR))
	{
		std::cout << input[0] << " XOR " << input[1] << " XOR " << input[2] << " = ";
		std::cout << xorNetwork.feedForward(input) << std::endl;
	}

	std::cout << "\n";
	
	for (const auto& input : keys(trainDataXNOR))
	{
		std::cout << input[0] << " XNOR " << input[1] << " XNOR " << input[2] << " = ";
		std::cout << xnorNetwork.feedForward(input) << std::endl;
	}
	
	return EXIT_SUCCESS;
}
