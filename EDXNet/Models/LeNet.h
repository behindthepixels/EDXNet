#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class LeNet
		{
		public:
			Symbol* input;
			Symbol* labels;

			Symbol* conv1Weights;
			Symbol* conv1Biases;
			Symbol* conv2Weights;
			Symbol* conv2Biases;
			Symbol* fc1Weights;
			Symbol* fc1Biases;
			Symbol* fc2Weights;
			Symbol* fc2Biases;
			Symbol* fc3Weights;
			Symbol* fc3Biases;

			Symbol* conv1;
			Symbol* relu1;
			Symbol* pool1;

			Symbol* conv2;
			Symbol* relu2;
			Symbol* pool2;

			Symbol* fullyConnected1;
			Symbol* relu3;

			Symbol* fullyConnected2;
			Symbol* relu4;

			Symbol* fullyConnected3;
			Symbol* leNet;

		public:
			// Serialization operator
			friend Stream& operator << (Stream& stream, LeNet& A);
			friend Stream& operator >> (Stream& stream, LeNet& A);

			static LeNet CreateForInference(const TCHAR* weightsPath);
			static LeNet CreateForTraining(const Tensorf& data, const Tensorf& labels);
		};

	}
}