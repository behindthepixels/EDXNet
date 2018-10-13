#include "VGG19.h"

#include "../Operators/Constant.h"
#include "../Operators/Variable.h"
#include "../Operators/FullyConnected.h"
#include "../Operators/Convolution.h"
#include "../Operators/Relu.h"
#include "../Operators/Softmax.h"
#include "../Operators/Predictor.h"

#include "Windows/FileStream.h"

namespace EDX
{
	namespace DeepLearning
	{
		template<typename PoolingOpType>
		void VGG19::CreateLayers()
		{
			// conv 64x3x3
			conv1_1 = NeuralNet::Create<Convolution>(input, weights.conv1_1_Weights, weights.conv1_1_Biases, TensorShape({ 3, 3 }), 64, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu1_1 = NeuralNet::Create<Relu>(conv1_1);
			conv1_2 = NeuralNet::Create<Convolution>(relu1_1, weights.conv1_2_Weights, weights.conv1_2_Biases, TensorShape({ 3, 3 }), 64, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu1_2 = NeuralNet::Create<Relu>(conv1_2);
			pool1 = NeuralNet::Create<PoolingOpType>(relu1_2, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			// conv 128x3x3
			conv2_1 = NeuralNet::Create<Convolution>(pool1, weights.conv2_1_Weights, weights.conv2_1_Biases, TensorShape({ 3, 3 }), 128, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu2_1 = NeuralNet::Create<Relu>(conv2_1);
			conv2_2 = NeuralNet::Create<Convolution>(relu2_1, weights.conv2_2_Weights, weights.conv2_2_Biases, TensorShape({ 3, 3 }), 128, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu2_2 = NeuralNet::Create<Relu>(conv2_2);
			pool2 = NeuralNet::Create<PoolingOpType>(relu2_2, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			// conv 256x3x3
			conv3_1 = NeuralNet::Create<Convolution>(pool2, weights.conv3_1_Weights, weights.conv3_1_Biases, TensorShape({ 3, 3 }), 256, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu3_1 = NeuralNet::Create<Relu>(conv3_1);
			conv3_2 = NeuralNet::Create<Convolution>(relu3_1, weights.conv3_2_Weights, weights.conv3_2_Biases, TensorShape({ 3, 3 }), 256, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu3_2 = NeuralNet::Create<Relu>(conv3_2);
			conv3_3 = NeuralNet::Create<Convolution>(relu3_2, weights.conv3_3_Weights, weights.conv3_3_Biases, TensorShape({ 3, 3 }), 256, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu3_3 = NeuralNet::Create<Relu>(conv3_3);
			conv3_4 = NeuralNet::Create<Convolution>(relu3_3, weights.conv3_4_Weights, weights.conv3_4_Biases, TensorShape({ 3, 3 }), 256, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu3_4 = NeuralNet::Create<Relu>(conv3_4);
			pool3 = NeuralNet::Create<PoolingOpType>(relu3_4, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			// conv 512x3x3
			conv4_1 = NeuralNet::Create<Convolution>(pool3, weights.conv4_1_Weights, weights.conv4_1_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu4_1 = NeuralNet::Create<Relu>(conv4_1);
			conv4_2 = NeuralNet::Create<Convolution>(relu4_1, weights.conv4_2_Weights, weights.conv4_2_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu4_2 = NeuralNet::Create<Relu>(conv4_2);
			conv4_3 = NeuralNet::Create<Convolution>(relu4_2, weights.conv4_3_Weights, weights.conv4_3_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu4_3 = NeuralNet::Create<Relu>(conv4_3);
			conv4_4 = NeuralNet::Create<Convolution>(relu4_3, weights.conv4_4_Weights, weights.conv4_4_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu4_4 = NeuralNet::Create<Relu>(conv4_4);
			pool4 = NeuralNet::Create<PoolingOpType>(relu4_4, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			// conv 512_2x3x3
			conv5_1 = NeuralNet::Create<Convolution>(pool4, weights.conv5_1_Weights, weights.conv5_1_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu5_1 = NeuralNet::Create<Relu>(conv5_1);
			conv5_2 = NeuralNet::Create<Convolution>(relu5_1, weights.conv5_2_Weights, weights.conv5_2_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu5_2 = NeuralNet::Create<Relu>(conv5_2);
			conv5_3 = NeuralNet::Create<Convolution>(relu5_2, weights.conv5_3_Weights, weights.conv5_3_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu5_3 = NeuralNet::Create<Relu>(conv5_3);
			conv5_4 = NeuralNet::Create<Convolution>(relu5_3, weights.conv5_4_Weights, weights.conv5_4_Biases, TensorShape({ 3, 3 }), 512, TensorShape({ 1, 1 }), TensorShape({ 1, 1 }));
			relu5_4 = NeuralNet::Create<Relu>(conv5_4);
			pool5 = NeuralNet::Create<PoolingOpType>(relu5_4, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			if (weights.withFullyConnected)
			{
				fullyConnected1 = NeuralNet::Create<FullyConnected>(pool5, weights.fc1_Weights, weights.fc1_Biases, 4096);
				reluFC1 = NeuralNet::Create<Relu>(fullyConnected1);

				fullyConnected2 = NeuralNet::Create<FullyConnected>(reluFC1, weights.fc2_Weights, weights.fc2_Biases, 4096);
				reluFC2 = NeuralNet::Create<Relu>(fullyConnected2);

				fullyConnected3 = NeuralNet::Create<FullyConnected>(reluFC2, weights.fc3_Weights, weights.fc3_Biases, 1000);

				loss = NeuralNet::Create<Predictor>(fullyConnected3);
			}
		}

		VGG19 VGG19::CreateForInference(const TCHAR* weightsPath,
			const bool withFullyConnected,
			const bool isInputVariable,
			const PoolingType poolingType)
		{
			VGG19Weights weights(withFullyConnected);
			weights.CreateConstant();

			FileStream stream(weightsPath, FileMode::Open);
			stream >> weights;
			stream.Close();

			return CreateForInference(weights, isInputVariable);
		}

		VGG19 VGG19::CreateForInference(const VGG19Weights& weights,
			const bool isInputVariable,
			const PoolingType poolingType)
		{
			VGG19 net;

			net.input = isInputVariable ?
				NeuralNet::Create<Variable>() :
				NeuralNet::Create<Constant>();

			// Weights have to be pre-initialized
			net.weights = weights;

			if (poolingType == PoolingType::Max)
				net.CreateLayers<MaxPooling>();
			else if (poolingType == PoolingType::Average)
				net.CreateLayers<AvgPooling>();

			return net;
		}

		VGG19 VGG19::CreateForTraining(const Tensorf& data, const Tensorf& labels, const PoolingType poolingType)
		{
			VGG19 net;

			net.input = NeuralNet::Create<Constant>(data);
			net.labels = NeuralNet::Create<Constant>(labels);

			net.weights.withFullyConnected = true;
			net.weights.CreateVariable();

			if (poolingType == PoolingType::Max)
				net.CreateLayers<MaxPooling>();
			else if (poolingType == PoolingType::Average)
				net.CreateLayers<AvgPooling>();

			return net;
		}

		// Serialization operator
		Stream& operator << (Stream& stream, VGG19& net)
		{
			// Save weight
			stream << net.weights;

			return stream;
		}

		Stream& operator >> (Stream& stream, VGG19& net)
		{
			// Load weights
			stream >> net.weights;

			return stream;
		}

		Stream& operator << (Stream& stream, VGG19Weights& weights)
		{
			stream << weights.conv1_1_Weights->GetOutput();
			stream << weights.conv1_1_Biases->GetOutput();
			stream << weights.conv1_2_Weights->GetOutput();
			stream << weights.conv1_2_Biases->GetOutput();
			stream << weights.conv2_1_Weights->GetOutput();
			stream << weights.conv2_1_Biases->GetOutput();
			stream << weights.conv2_2_Weights->GetOutput();
			stream << weights.conv2_2_Biases->GetOutput();
			stream << weights.conv3_1_Weights->GetOutput();
			stream << weights.conv3_1_Biases->GetOutput();
			stream << weights.conv3_2_Weights->GetOutput();
			stream << weights.conv3_2_Biases->GetOutput();
			stream << weights.conv3_3_Weights->GetOutput();
			stream << weights.conv3_3_Biases->GetOutput();
			stream << weights.conv3_4_Weights->GetOutput();
			stream << weights.conv3_4_Biases->GetOutput();
			stream << weights.conv4_1_Weights->GetOutput();
			stream << weights.conv4_1_Biases->GetOutput();
			stream << weights.conv4_2_Weights->GetOutput();
			stream << weights.conv4_2_Biases->GetOutput();
			stream << weights.conv4_3_Weights->GetOutput();
			stream << weights.conv4_3_Biases->GetOutput();
			stream << weights.conv4_4_Weights->GetOutput();
			stream << weights.conv4_4_Biases->GetOutput();
			stream << weights.conv5_1_Weights->GetOutput();
			stream << weights.conv5_1_Biases->GetOutput();
			stream << weights.conv5_2_Weights->GetOutput();
			stream << weights.conv5_2_Biases->GetOutput();
			stream << weights.conv5_3_Weights->GetOutput();
			stream << weights.conv5_3_Biases->GetOutput();
			stream << weights.conv5_4_Weights->GetOutput();
			stream << weights.conv5_4_Biases->GetOutput();

			if (weights.withFullyConnected)
			{
				stream << weights.fc1_Weights->GetOutput();
				stream << weights.fc1_Biases->GetOutput();
				stream << weights.fc2_Weights->GetOutput();
				stream << weights.fc2_Biases->GetOutput();
				stream << weights.fc3_Weights->GetOutput();
				stream << weights.fc3_Biases->GetOutput();
			}

			return stream;
		}

		Stream& operator >> (Stream& stream, VGG19Weights& weights)
		{
			stream >> weights.conv1_1_Weights->GetOutput();
			stream >> weights.conv1_1_Biases->GetOutput();
			stream >> weights.conv1_2_Weights->GetOutput();
			stream >> weights.conv1_2_Biases->GetOutput();
			stream >> weights.conv2_1_Weights->GetOutput();
			stream >> weights.conv2_1_Biases->GetOutput();
			stream >> weights.conv2_2_Weights->GetOutput();
			stream >> weights.conv2_2_Biases->GetOutput();
			stream >> weights.conv3_1_Weights->GetOutput();
			stream >> weights.conv3_1_Biases->GetOutput();
			stream >> weights.conv3_2_Weights->GetOutput();
			stream >> weights.conv3_2_Biases->GetOutput();
			stream >> weights.conv3_3_Weights->GetOutput();
			stream >> weights.conv3_3_Biases->GetOutput();
			stream >> weights.conv3_4_Weights->GetOutput();
			stream >> weights.conv3_4_Biases->GetOutput();
			stream >> weights.conv4_1_Weights->GetOutput();
			stream >> weights.conv4_1_Biases->GetOutput();
			stream >> weights.conv4_2_Weights->GetOutput();
			stream >> weights.conv4_2_Biases->GetOutput();
			stream >> weights.conv4_3_Weights->GetOutput();
			stream >> weights.conv4_3_Biases->GetOutput();
			stream >> weights.conv4_4_Weights->GetOutput();
			stream >> weights.conv4_4_Biases->GetOutput();
			stream >> weights.conv5_1_Weights->GetOutput();
			stream >> weights.conv5_1_Biases->GetOutput();
			stream >> weights.conv5_2_Weights->GetOutput();
			stream >> weights.conv5_2_Biases->GetOutput();
			stream >> weights.conv5_3_Weights->GetOutput();
			stream >> weights.conv5_3_Biases->GetOutput();
			stream >> weights.conv5_4_Weights->GetOutput();
			stream >> weights.conv5_4_Biases->GetOutput();

			if (weights.withFullyConnected)
			{
				stream >> weights.fc1_Weights->GetOutput();
				stream >> weights.fc1_Biases->GetOutput();
				stream >> weights.fc2_Weights->GetOutput();
				stream >> weights.fc2_Biases->GetOutput();
				stream >> weights.fc3_Weights->GetOutput();
				stream >> weights.fc3_Biases->GetOutput();
			}

			return stream;
		}

		void VGG19Weights::CreateConstant()
		{
			conv1_1_Weights = NeuralNet::Create<Constant>();
			conv1_1_Biases = NeuralNet::Create<Constant>();
			conv1_2_Weights = NeuralNet::Create<Constant>();
			conv1_2_Biases = NeuralNet::Create<Constant>();
			conv2_1_Weights = NeuralNet::Create<Constant>();
			conv2_1_Biases = NeuralNet::Create<Constant>();
			conv2_2_Weights = NeuralNet::Create<Constant>();
			conv2_2_Biases = NeuralNet::Create<Constant>();
			conv3_1_Weights = NeuralNet::Create<Constant>();
			conv3_1_Biases = NeuralNet::Create<Constant>();
			conv3_2_Weights = NeuralNet::Create<Constant>();
			conv3_2_Biases = NeuralNet::Create<Constant>();
			conv3_3_Weights = NeuralNet::Create<Constant>();
			conv3_3_Biases = NeuralNet::Create<Constant>();
			conv3_4_Weights = NeuralNet::Create<Constant>();
			conv3_4_Biases = NeuralNet::Create<Constant>();
			conv4_1_Weights = NeuralNet::Create<Constant>();
			conv4_1_Biases = NeuralNet::Create<Constant>();
			conv4_2_Weights = NeuralNet::Create<Constant>();
			conv4_2_Biases = NeuralNet::Create<Constant>();
			conv4_3_Weights = NeuralNet::Create<Constant>();
			conv4_3_Biases = NeuralNet::Create<Constant>();
			conv4_4_Weights = NeuralNet::Create<Constant>();
			conv4_4_Biases = NeuralNet::Create<Constant>();
			conv5_1_Weights = NeuralNet::Create<Constant>();
			conv5_1_Biases = NeuralNet::Create<Constant>();
			conv5_2_Weights = NeuralNet::Create<Constant>();
			conv5_2_Biases = NeuralNet::Create<Constant>();
			conv5_3_Weights = NeuralNet::Create<Constant>();
			conv5_3_Biases = NeuralNet::Create<Constant>();
			conv5_4_Weights = NeuralNet::Create<Constant>();
			conv5_4_Biases = NeuralNet::Create<Constant>();
		}

		void VGG19Weights::CreateVariable()
		{
			conv1_1_Weights = NeuralNet::Create<Variable>();
			conv1_1_Biases = NeuralNet::Create<Variable>();
			conv1_2_Weights = NeuralNet::Create<Variable>();
			conv1_2_Biases = NeuralNet::Create<Variable>();
			conv2_1_Weights = NeuralNet::Create<Variable>();
			conv2_1_Biases = NeuralNet::Create<Variable>();
			conv2_2_Weights = NeuralNet::Create<Variable>();
			conv2_2_Biases = NeuralNet::Create<Variable>();
			conv3_1_Weights = NeuralNet::Create<Variable>();
			conv3_1_Biases = NeuralNet::Create<Variable>();
			conv3_2_Weights = NeuralNet::Create<Variable>();
			conv3_2_Biases = NeuralNet::Create<Variable>();
			conv3_3_Weights = NeuralNet::Create<Variable>();
			conv3_3_Biases = NeuralNet::Create<Variable>();
			conv3_4_Weights = NeuralNet::Create<Variable>();
			conv3_4_Biases = NeuralNet::Create<Variable>();
			conv4_1_Weights = NeuralNet::Create<Variable>();
			conv4_1_Biases = NeuralNet::Create<Variable>();
			conv4_2_Weights = NeuralNet::Create<Variable>();
			conv4_2_Biases = NeuralNet::Create<Variable>();
			conv4_3_Weights = NeuralNet::Create<Variable>();
			conv4_3_Biases = NeuralNet::Create<Variable>();
			conv4_4_Weights = NeuralNet::Create<Variable>();
			conv4_4_Biases = NeuralNet::Create<Variable>();
			conv5_1_Weights = NeuralNet::Create<Variable>();
			conv5_1_Biases = NeuralNet::Create<Variable>();
			conv5_2_Weights = NeuralNet::Create<Variable>();
			conv5_2_Biases = NeuralNet::Create<Variable>();
			conv5_3_Weights = NeuralNet::Create<Variable>();
			conv5_3_Biases = NeuralNet::Create<Variable>();
			conv5_4_Weights = NeuralNet::Create<Variable>();
			conv5_4_Biases = NeuralNet::Create<Variable>();
		}
	}
}