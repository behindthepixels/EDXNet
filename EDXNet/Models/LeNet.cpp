#include "LeNet.h"

#include "../Operators/Constant.h"
#include "../Operators/Variable.h"
#include "../Operators/FullyConnected.h"
#include "../Operators/Convolution.h"
#include "../Operators/Pooling.h"
#include "../Operators/Relu.h"
#include "../Operators/Softmax.h"
#include "../Operators/Predictor.h"

#include "Windows/FileStream.h"

namespace EDX
{
	namespace DeepLearning
	{
		LeNet LeNet::CreateForInference(const TCHAR* weightsPath, const Tensorf& inputTensor)
		{
			LeNet net;

			net.input = NeuralNet::Create<Constant>(inputTensor);

			net.conv1Weights = NeuralNet::Create<Constant>();
			net.conv1Biases = NeuralNet::Create<Constant>();
			net.conv2Weights = NeuralNet::Create<Constant>();
			net.conv2Biases = NeuralNet::Create<Constant>();
			net.fc1Weights = NeuralNet::Create<Constant>();
			net.fc1Biases = NeuralNet::Create<Constant>();
			net.fc2Weights = NeuralNet::Create<Constant>();
			net.fc2Biases = NeuralNet::Create<Constant>();
			net.fc3Weights = NeuralNet::Create<Constant>();
			net.fc3Biases = NeuralNet::Create<Constant>();

			net.conv1 = NeuralNet::Create<Convolution>(net.input, net.conv1Weights, net.conv1Biases, TensorShape({ 5, 5 }), 6, TensorShape({ 1, 1 }), TensorShape({ 2, 2 }));
			net.relu1 = NeuralNet::Create<Relu>(net.conv1);
			net.pool1 = NeuralNet::Create<MaxPooling>(net.relu1, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			net.conv2 = NeuralNet::Create<Convolution>(net.pool1, net.conv2Weights, net.conv2Biases, TensorShape({ 5, 5 }), 16, TensorShape({ 1, 1 }), TensorShape({ 2, 2 }));
			net.relu2 = NeuralNet::Create<Relu>(net.conv2);
			net.pool2 = NeuralNet::Create<MaxPooling>(net.relu2, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			net.fullyConnected1 = NeuralNet::Create<FullyConnected>(net.pool2, net.fc1Weights, net.fc1Biases, 120);
			net.relu3 = NeuralNet::Create<Relu>(net.fullyConnected1);

			net.fullyConnected2 = NeuralNet::Create<FullyConnected>(net.relu3, net.fc2Weights, net.fc2Biases, 84);
			net.relu4 = NeuralNet::Create<Relu>(net.fullyConnected2);

			net.fullyConnected3 = NeuralNet::Create<FullyConnected>(net.relu4, net.fc3Weights, net.fc3Biases, 10);
			net.leNet = NeuralNet::Create<Predictor>(net.fullyConnected3);

			FileStream stream(weightsPath, FileMode::Open);
			stream >> net;
			stream.Close();

			return net;
		}

		LeNet LeNet::CreateForTraining(const Tensorf& data, const Tensorf& labels)
		{
			LeNet net;

			net.input = NeuralNet::Create<Constant>(data);
			net.labels = NeuralNet::Create<Constant>(labels);

			net.conv1Weights = NeuralNet::Create<Variable>();
			net.conv1Biases = NeuralNet::Create<Variable>();
			net.conv2Weights = NeuralNet::Create<Variable>();
			net.conv2Biases = NeuralNet::Create<Variable>();
			net.fc1Weights = NeuralNet::Create<Variable>();
			net.fc1Biases = NeuralNet::Create<Variable>();
			net.fc2Weights = NeuralNet::Create<Variable>();
			net.fc2Biases = NeuralNet::Create<Variable>();
			net.fc3Weights = NeuralNet::Create<Variable>();
			net.fc3Biases = NeuralNet::Create<Variable>();

			net.conv1 = NeuralNet::Create<Convolution>(net.input, net.conv1Weights, net.conv1Biases, TensorShape({ 5, 5 }), 6, TensorShape({ 1, 1 }), TensorShape({ 2, 2 }));
			net.relu1 = NeuralNet::Create<Relu>(net.conv1);
			net.pool1 = NeuralNet::Create<MaxPooling>(net.relu1, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			net.conv2 = NeuralNet::Create<Convolution>(net.pool1, net.conv2Weights, net.conv2Biases, TensorShape({ 5, 5 }), 16, TensorShape({ 1, 1 }), TensorShape({ 2, 2 }));
			net.relu2 = NeuralNet::Create<Relu>(net.conv2);
			net.pool2 = NeuralNet::Create<MaxPooling>(net.relu2, TensorShape({ 2, 2 }), TensorShape({ 2, 2 }), TensorShape({ 0, 0 }));

			net.fullyConnected1 = NeuralNet::Create<FullyConnected>(net.pool2, net.fc1Weights, net.fc1Biases, 120);
			net.relu3 = NeuralNet::Create<Relu>(net.fullyConnected1);

			net.fullyConnected2 = NeuralNet::Create<FullyConnected>(net.relu3, net.fc2Weights, net.fc2Biases, 84);
			net.relu4 = NeuralNet::Create<Relu>(net.fullyConnected2);

			net.fullyConnected3 = NeuralNet::Create<FullyConnected>(net.relu4, net.fc3Weights, net.fc3Biases, 10);
			net.leNet = NeuralNet::Create<Softmax>(net.fullyConnected3, net.labels);

			return net;
		}

		// Serialization operator
		Stream& operator << (Stream& stream, LeNet& A)
		{
			// Save weights
			stream << A.conv1Weights->GetOutput();
			stream << A.conv1Biases->GetOutput();
			stream << A.conv2Weights->GetOutput();
			stream << A.conv2Biases->GetOutput();
			stream << A.fc1Weights->GetOutput();
			stream << A.fc1Biases->GetOutput();
			stream << A.fc2Weights->GetOutput();
			stream << A.fc2Biases->GetOutput();
			stream << A.fc3Weights->GetOutput();
			stream << A.fc3Biases->GetOutput();

			return stream;
		}

		Stream& operator >> (Stream& stream, LeNet& A)
		{
			// Load weights.

			stream >> A.conv1Weights->GetOutput();
			stream >> A.conv1Biases->GetOutput();
			stream >> A.conv2Weights->GetOutput();
			stream >> A.conv2Biases->GetOutput();
			stream >> A.fc1Weights->GetOutput();
			stream >> A.fc1Biases->GetOutput();
			stream >> A.fc2Weights->GetOutput();
			stream >> A.fc2Biases->GetOutput();
			stream >> A.fc3Weights->GetOutput();
			stream >> A.fc3Biases->GetOutput();

			return stream;
		}
	}
}