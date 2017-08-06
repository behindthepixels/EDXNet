#pragma once

#include "../Core/NeuralNet.h"
#include "../Operators/Pooling.h"

namespace EDX
{
	namespace DeepLearning
	{
		struct VGG19Weights
		{
			VGG19Weights(bool _withFullyConnected = true)
				: withFullyConnected(_withFullyConnected)
			{
			}

			bool withFullyConnected;
			Symbol* conv1_1_Weights;
			Symbol* conv1_1_Biases;
			Symbol* conv1_2_Weights;
			Symbol* conv1_2_Biases;
			Symbol* conv2_1_Weights;
			Symbol* conv2_1_Biases;
			Symbol* conv2_2_Weights;
			Symbol* conv2_2_Biases;
			Symbol* conv3_1_Weights;
			Symbol* conv3_1_Biases;
			Symbol* conv3_2_Weights;
			Symbol* conv3_2_Biases;
			Symbol* conv3_3_Weights;
			Symbol* conv3_3_Biases;
			Symbol* conv3_4_Weights;
			Symbol* conv3_4_Biases;
			Symbol* conv4_1_Weights;
			Symbol* conv4_1_Biases;
			Symbol* conv4_2_Weights;
			Symbol* conv4_2_Biases;
			Symbol* conv4_3_Weights;
			Symbol* conv4_3_Biases;
			Symbol* conv4_4_Weights;
			Symbol* conv4_4_Biases;
			Symbol* conv5_1_Weights;
			Symbol* conv5_1_Biases;
			Symbol* conv5_2_Weights;
			Symbol* conv5_2_Biases;
			Symbol* conv5_3_Weights;
			Symbol* conv5_3_Biases;
			Symbol* conv5_4_Weights;
			Symbol* conv5_4_Biases;
			Symbol* fc1_Weights;
			Symbol* fc1_Biases;
			Symbol* fc2_Weights;
			Symbol* fc2_Biases;
			Symbol* fc3_Weights;
			Symbol* fc3_Biases;
			
		public:

			void CreateConstant();
			void CreateVariable();

			// Serialization operator
			friend Stream& operator << (Stream& stream, VGG19Weights& weights);
			friend Stream& operator >> (Stream& stream, VGG19Weights& weights);
		};

		class VGG19
		{
		public:
			Symbol* input;
			Symbol* labels;

			// Model weights
			VGG19Weights weights;

			// conv 64x3x3
			Symbol* conv1_1;
			Symbol* relu1_1;
			Symbol* conv1_2;
			Symbol* relu1_2;
			Symbol* pool1;

			// conv 128x3x3
			Symbol* conv2_1;
			Symbol* relu2_1;
			Symbol* conv2_2;
			Symbol* relu2_2;
			Symbol* pool2;

			// conv 256x3x3;
			Symbol* conv3_1;
			Symbol* relu3_1;
			Symbol* conv3_2;
			Symbol* relu3_2;
			Symbol* conv3_3;
			Symbol* relu3_3;
			Symbol* conv3_4;
			Symbol* relu3_4;
			Symbol* pool3;

			// conv 512x3x3;
			Symbol* conv4_1;
			Symbol* relu4_1;
			Symbol* conv4_2;
			Symbol* relu4_2;
			Symbol* conv4_3;
			Symbol* relu4_3;
			Symbol* conv4_4;
			Symbol* relu4_4;
			Symbol* pool4;

			// conv 512_2x3x3
			Symbol* conv5_1;
			Symbol* relu5_1;
			Symbol* conv5_2;
			Symbol* relu5_2;
			Symbol* conv5_3;
			Symbol* relu5_3;
			Symbol* conv5_4;
			Symbol* relu5_4;
			Symbol* pool5;

			Symbol* fullyConnected1;
			Symbol* reluFC1;

			Symbol* fullyConnected2;
			Symbol* reluFC2;

			Symbol* fullyConnected3;

			Symbol* loss;


		public:
			// Serialization operator
			friend Stream& operator << (Stream& stream, VGG19& net);
			friend Stream& operator >> (Stream& stream, VGG19& net);

			template<typename PoolingOpType>
			void CreateLayers();

			static VGG19 CreateForInference(const TCHAR* weightsPath,
				const bool withFullyConnected = true,
				const bool isInputVariable = false,
				const PoolingType poolingType = PoolingType::Max);
			static VGG19 CreateForTraining(const Tensorf& data, const Tensorf& labels, const PoolingType poolingType = PoolingType::Max);

			static VGG19 CreateForInference(const VGG19Weights& weights,
				const bool isInputVariable = false,
				const PoolingType poolingType = PoolingType::Max);
		};

	}
}