
#include "Core/EDXNet.h"

using namespace EDX;
using namespace DeepLearning;

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

	std::cout << Tensorf::Sum(Tensorf(Abs(result - correctResult))) << "\n";

	Tensorf& dataGrad = symbolsToEvaluate[0]->GetOutput(data->GetGradientIndex());
	Tensorf& weightsGrad = symbolsToEvaluate[1]->GetOutput(weights->GetGradientIndex());
	Tensorf& biasesGrad = symbolsToEvaluate[2]->GetOutput(biases->GetGradientIndex());

	Tensorf upperGrads = Ones(result.Shape());
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

	std::cout << Tensorf::Sum(Tensorf(Abs(dataGrad - dataNumericalGrad))) << "\n";
	std::cout << Tensorf::Sum(Tensorf(Abs(weightsGrad - numericalWeightsGrad))) << "\n";
	std::cout << Tensorf::Sum(Tensorf(Abs(biasesGrad - numericalBiasesGrad))) << "\n";

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

	std::cout << Tensorf::Sum(Tensorf(Abs(result - correctResult))) << "\n";

	Tensorf& dataGrad = symbolsToEvaluate[0]->GetOutput(data->GetGradientIndex());
	Tensorf& weightsGrad = symbolsToEvaluate[1]->GetOutput(weights->GetGradientIndex());
	Tensorf& biasesGrad = symbolsToEvaluate[2]->GetOutput(biases->GetGradientIndex());

	Tensorf upperGrads = Ones(result.Shape());
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

	std::cout << Tensorf::Sum(Abs(dataGrad - dataNumericalGrad)) << "\n";
	std::cout << Tensorf::Sum(Abs(weightsGrad - numericalWeightsGrad)) << "\n";
	std::cout << Tensorf::Sum(Abs(biasesGrad - numericalBiasesGrad)) << "\n";

	NeuralNet::Release();

	return true;
}

bool TestReluCUDA()
{
	Tensorf input = Tensorf::LinSpace(-0.5, 0.5, 12).Reshape(3, 4);

	Symbol* x = NeuralNet::Create<Variable>(input);
	Symbol* relu = NeuralNet::Create<Relu>(x);

	NeuralNet net(relu, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ x });
	symbolsToEvaluate.Add(relu); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf result = relu->GetOutput();
	Tensorf correctResult = Tensorf({ { 0.0f, 0.0f, 0.0f, 0.0f, },
									{ 0.0f, 0.0f, 0.04545455f, 0.13636364f, },
									{ 0.22727273f, 0.31818182f, 0.40909091f, 0.5f, } });

	std::cout << Tensorf::Sum(Tensorf(Abs(result - correctResult))) << "\n";

	Tensorf& dx = symbolsToEvaluate[0]->GetOutput(0);

	Tensorf upperGrads = Ones(result.Shape());
	Tensorf dxNumerical = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ relu });
			return relu->GetOutput();
		},
		x, upperGrads);

	std::cout << Tensorf::Sum(Tensorf(Abs(dx - dxNumerical))) << "\n";

	NeuralNet::Release();

	return true;
}


void TestPoolingCUDA()
{
	TensorShape xShape = { 2,3,4,4 };
	Tensorf input = Tensorf::LinSpace(-0.3f, 0.4f, Algorithm::Accumulate(xShape, 1, Algorithm::Multiply<>())).Reshape(xShape);

	Symbol* x = NeuralNet::Create<Variable>(input);
	Symbol* pooling = NeuralNet::Create<MaxPooling>(x, TensorShape({ 2,2 }), TensorShape({ 2,2 }), TensorShape({ 0,0 }));

	NeuralNet net(pooling, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ x });
	symbolsToEvaluate.Add(pooling); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf& result = pooling->GetOutput();
	Tensorf correctResult = Tensorf({ { { { -0.26315789f, -0.24842105f },
											{ -0.20421053f, -0.18947368f } },
											{ { -0.14526316f, -0.13052632f },
											{ -0.08631579f, -0.07157895f } },
											{ { -0.02736842f, -0.01263158f },
											{ 0.03157895f, 0.04631579f } } },
											{ { { 0.09052632f, 0.10526316f },
											{ 0.14947368f, 0.16421053f } },
											{ { 0.20842105f, 0.22315789f },
											{ 0.26736842f, 0.28210526f } },
											{ { 0.32631579f, 0.34105263f },
											{ 0.38526316f, 0.4f } } } });

	std::cout << Tensorf::Sum(Abs(result - correctResult)) << "\n";

	Tensorf& dx = symbolsToEvaluate[0]->GetOutput(0);

	Tensorf upperGrads = Ones(result.Shape());
	Tensorf dxNumerical = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ pooling });
			return pooling->GetOutput();
		},
		x, upperGrads);

	std::cout << Tensorf::Sum(Abs(dx - dxNumerical)) << "\n";

	NeuralNet::Release();
}


void TestSoftmaxCUDA()
{
	const int numClasses = 10;
	const int numInputs = 50;
	Tensorf dataX = Scalar(0.001f) * Tensorf::RandomNormalDistribution(1.0, numInputs, numClasses);
	Tensorf dataY = Tensorf::RandomInt(numClasses, numInputs);

	Symbol* x = NeuralNet::Create<Variable>(dataX);
	Symbol* y = NeuralNet::Create<Constant>(dataY);
	Symbol* softmax = NeuralNet::Create<Softmax>(x, y);

	NeuralNet net(softmax, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ x });
	symbolsToEvaluate.Add(softmax); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf result = softmax->GetOutput();
	Tensorf correctResult = 2.3f;

	std::cout << Tensorf::Sum(Abs(result - correctResult)) << "\n";

	Tensorf& dx = symbolsToEvaluate[0]->GetOutput(0);

	Tensorf upperGrads = Ones(result.Shape());
	Tensorf dxNumerical = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ softmax });
			return softmax->GetOutput();
		},
		x, upperGrads);

	std::cout << Tensorf::Sum(Abs(dx - dxNumerical)) << "\n";

	NeuralNet::Release();
}

void TestCUDA()
{
	TestFullyConnectedCUDA();
	TestConvolutionCUDA();
	TestReluCUDA();
	TestPoolingCUDA();
	TestSoftmaxCUDA();


	{
		Tensorf A = { 1,2,3,4,5,8,3,1,4 };
		Tensorf B = NestedInitializerList<float, 2>({ {1},{2},{3},{4},{5} });

		Tensorf C = A + B + A + A * B;

		std::cout << C << "\n";
	}

	{
		Tensorf A = { 9,2,3,4,5 };
		Tensorf B = { 9,2,3,4,5 };

		A += Exp(B);

		std::cout << A << "\n";
	}

	{
		Tensorf A = { { 9,2,3,4,5 } };
		Tensorf B = NestedInitializerList<float, 2>({ {1},{2},{3},{4},{5} });

		Tensorf C = Dot((A + B) * Sum(Tensorf::LinSpace(0, 128, 128)), B) * B;

		std::cout << C << "\n";
	}

	{
		Tensorf A = Tensorf::LinSpace(0, 40960, 40960);
		Tensorf std = StandardDeviation(A);

		std::cout << std << "\n";
	}

	{
		Tensorf A = Tensorf::ArrayRange(0, 1000, 1.0);
		A.Reshape(5, 10, 4, 5);

		Tensorf sum = Sum(A, { 0, 2 }, true);

		std::cout << sum << "\n";
	}

	{
		Tensorf A = { {9,2,3,4,5} };
		Tensorf B = NestedInitializerList<float, 2>({ {9},{2},{3},{4},{5} });

		Tensorf C = Tensorf::Dot(B, A);

		std::cout << C << "\n";
	}

	{
		Tensorf A = Tensorf::LinSpace(0, 10, 10, true);
		auto exp = StandardDeviation(A);
		Tensorf std = exp;
		exp.Backward(Ones(exp.Shape()));

		Tensorf diff = A.GetGrad();
		std::cout << diff << "\n";

		Tensorf numericalDiff = NumericalGradientEval(exp, A);
		std::cout << numericalDiff << "\n";
	}

	{
		Tensorf A = Tensorf::ArrayRange(10, true).Reshape(2, 5);
		Tensorf B = Tensorf::ArrayRange(10, true).Reshape(5, 2);
		Tensorf C({ 7 }, true);

		auto exp = Dot(Dot(B, A) * Log(C), Sin(B)) * Sum(Square(Cos(A * C))) + C;
		Tensorf results = exp;

		exp.Backward(Ones(exp.Shape()));

		Tensorf diff = A.GetGrad();
		std::cout << diff << "\n";

		Tensorf numericalDiff = NumericalGradientEval(exp, A);
		std::cout << numericalDiff << "\n";
	}
}