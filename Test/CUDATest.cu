
#include "Core/EDXNet.h"

using namespace EDX;
using namespace DeepLearning;
using namespace Algorithm;

void TestFullyConnectedCUDA()
{
	int numInputs = 2;
	int numHidden = 3;
	TensorShape inputShape = { 4, 5, 6 };

	int inputSize = numInputs * Algorithm::Accumulate(inputShape, 1, Algorithm::Multiply<>());
	int weightSize = numHidden * Algorithm::Accumulate(inputShape, 1, Algorithm::Multiply<>());

	Tensorf x = Tensorf::LinSpace(-0.1f, 0.5f, inputSize).Reshape(numInputs, inputSize / numInputs);
	Tensorf w = Tensorf::LinSpace(-0.2f, 0.3f, weightSize).Reshape(inputSize / numInputs, numHidden);
	Tensorf b = Tensorf::LinSpace(-0.3f, 0.1f, numHidden);

	Tensorf r = Tensorf::Dot(x, w);

	Symbol* data = NeuralNet::Create<Variable>(x);
	Symbol* weights = NeuralNet::Create<Variable>(w);
	Symbol* biases = NeuralNet::Create<Variable>(b);

	Symbol* fullyConnected = NeuralNet::Create<FullyConnected>(data, weights, biases, numHidden);

	NeuralNet net(fullyConnected, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ data, weights, biases });
	symbolsToEvaluate.Add(fullyConnected); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf result = fullyConnected->GetOutput();

	Tensorf correctResult = Tensorf({ { 1.49834967f, 1.70660132f, 1.91485297f },
										{ 3.25553199f, 3.5141327f, 3.77273342f } });

	std::cout << Tensorf::Sum(Tensorf(Tensorf::Abs(result - correctResult))) << "\n";

	Tensorf dataGrad = symbolsToEvaluate[0]->GetOutput(data->GetGradientIndex());
	Tensorf weightsGrad = symbolsToEvaluate[1]->GetOutput(weights->GetGradientIndex());
	Tensorf biasesGrad = symbolsToEvaluate[2]->GetOutput(biases->GetGradientIndex());

	Tensorf upperGrads = Tensorf::Ones(result.Shape());
	Tensorf dataNumericalGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ fullyConnected });
		return fullyConnected->GetOutput();
	},
		data, upperGrads);
	Tensorf numericalWeightsGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ fullyConnected });
		return fullyConnected->GetOutput();
	},
		weights, upperGrads);
	Tensorf numericalBiasesGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ fullyConnected });
		return fullyConnected->GetOutput();
	},
		biases, upperGrads);

	std::cout << Tensorf::Sum(Tensorf(Tensorf::Abs(dataGrad - dataNumericalGrad))) << "\n";
	std::cout << Tensorf::Sum(Tensorf(Tensorf::Abs(weightsGrad - numericalWeightsGrad))) << "\n";
	std::cout << Tensorf::Sum(Tensorf(Tensorf::Abs(biasesGrad - numericalBiasesGrad))) << "\n";

	NeuralNet::Release();
}

bool TestConvolutionCUDA()
{
	TensorShape x_shape = { 2, 3, 4, 4 };
	TensorShape w_shape = { 3, 3, 4, 4 };

	int xSize = Algorithm::Accumulate(x_shape, 1, Algorithm::Multiply<>());
	int wSize = Algorithm::Accumulate(w_shape, 1, Algorithm::Multiply<>());

	Tensorf x = Tensorf::LinSpace(-0.1f, 0.5f, xSize).Reshape(x_shape);
	Tensorf w = Tensorf::LinSpace(-0.2f, 0.3f, wSize).Reshape(w_shape);
	Tensorf b = Tensorf::LinSpace(-0.1f, 0.2f, 3);

	Symbol* data = NeuralNet::Create<Variable>(x);
	Symbol* weights = NeuralNet::Create<Variable>(w);
	Symbol* biases = NeuralNet::Create<Variable>(b);

	Symbol* convolution = NeuralNet::Create<Convolution>(data, weights, biases, TensorShape({ 4, 4 }), 3, TensorShape({ 2, 2 }), TensorShape({ 1, 1 }));

	NeuralNet net(convolution, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ data, weights, biases });
	symbolsToEvaluate.Add(convolution); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf result = convolution->GetOutput();
	Tensorf correctResult = Tensorf({ { { { -0.08759809f, -0.10987781f },
			{-0.18387192f, -0.2109216f}},
			{{0.21027089f, 0.21661097f},
			{0.22847626f, 0.23004637f}},
			{{0.50813986f, 0.54309974f},
			{0.64082444f, 0.67101435f}}},
			{{{-0.98053589f, -1.03143541f},
			{-1.19128892f, -1.24695841f}},
			{{0.69108355f, 0.66880383f},
			{0.59480972f, 0.56776003f}},
			{{2.36270298f, 2.36904306f},
			{2.38090835f, 2.38247847f}}} });

	std::cout << Tensorf::Sum(Tensorf(Tensorf::Abs(result - correctResult))) << "\n";

	Tensorf dataGrad = symbolsToEvaluate[0]->GetOutput(data->GetGradientIndex());
	Tensorf weightsGrad = symbolsToEvaluate[1]->GetOutput(weights->GetGradientIndex());
	Tensorf biasesGrad = symbolsToEvaluate[2]->GetOutput(biases->GetGradientIndex());

	Tensorf upperGrads = Tensorf::Ones(result.Shape());
	Tensorf dataNumericalGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ convolution });
		return convolution->GetOutput();
	},
		data, upperGrads);
	Tensorf numericalWeightsGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ convolution });
		return convolution->GetOutput();
	},
		weights, upperGrads);
	Tensorf numericalBiasesGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ convolution });
		return convolution->GetOutput();
	},
		biases, upperGrads);

	std::cout << Tensorf::Sum(Tensorf::Abs(dataGrad - dataNumericalGrad)) << "\n";
	std::cout << Tensorf::Sum(Tensorf::Abs(weightsGrad - numericalWeightsGrad)) << "\n";
	std::cout << Tensorf::Sum(Tensorf::Abs(biasesGrad - numericalBiasesGrad)) << "\n";

	NeuralNet::Release();

	return true;
}

void TestCUDA()
{
	{
		Tensorf A = { 1,2,3,4,5,8,3,1,4 };
		Tensorf B = NestedInitializerList<float, 2>({ {1},{2},{3},{4},{5} });

		Tensorf C = A + B + A + A * B;

		float pHostC[45] = { 0 };
		cudaMemcpy((void*)pHostC, (void*)C.Data(), C.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	{
		Tensorf A = { 9,2,3,4,5 };
		Tensorf B = { 9,2,3,4,5 };

		A += Tensorf::Exp(B);

		float pHostA[5] = { 0 };
		cudaMemcpy((void*)pHostA, (void*)A.Data(), A.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	{
		Tensorf A = Tensorf::LinSpace(0, 40960, 40960);
		Tensorf sum = Tensorf::StandardDeviation(A);

		float pHost[1] = { 0 };
		cudaMemcpy((void*)pHost, (void*)sum.Data(), sum.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	{
		Tensorf A = Tensorf::ArrayRange(0, 1000);
		A.Reshape(5, 10, 4, 5);

		Tensorf sum = Tensorf::Sum(A, { 0, 2 }, true);

		float pHost[50] = { 0 };
		cudaMemcpy((void*)pHost, (void*)sum.Data(), sum.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	{
		Tensorf A = { {9,2,3,4,5} };
		Tensorf B = NestedInitializerList<float, 2>({ {9},{2},{3},{4},{5} });

		Tensorf C = Tensorf::Dot(B, A);

		float pHost[25] = { 0 };
		cudaMemcpy((void*)pHost, (void*)C.Data(), C.LinearSize() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	TestFullyConnectedCUDA();
	TestConvolutionCUDA();
}