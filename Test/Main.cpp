#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h>  

#include "Core/EDXNet.h"

#include "Containers/Map.h"
#include "Core/SmartPointer.h"
#include "Windows/Bitmap.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace EDX;
using namespace DeepLearning;
using namespace Algorithm;


void TestFullyConnected()
{
	int numInputs = 2;
	int numHidden = 3;
	Array<int> inputShape = { 4, 5, 6 };

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

	Tensorf correctResult = Tensorf({	{ 1.49834967f, 1.70660132f, 1.91485297f },
										{ 3.25553199f, 3.5141327f, 3.77273342f } });

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(result - correctResult)), 0.0f) << "\n";

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

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(dataGrad - dataNumericalGrad)), 0.0f) << "\n";
	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(weightsGrad - numericalWeightsGrad)), 0.0f) << "\n";
	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(biasesGrad - numericalBiasesGrad)), 0.0f) << "\n";

	NeuralNet::Release();
}

bool TestConvolution()
{
	Array<int> x_shape = { 2, 3, 4, 4 };
	Array<int> w_shape = { 3, 3, 4, 4 };

	int xSize = Algorithm::Accumulate(x_shape, 1, Algorithm::Multiply<>());
	int wSize = Algorithm::Accumulate(w_shape, 1, Algorithm::Multiply<>());

	Tensorf x = Tensorf::LinSpace(-0.1f, 0.5f, xSize).Reshape(x_shape);
	Tensorf w = Tensorf::LinSpace(-0.2f, 0.3f, wSize).Reshape(w_shape);
	Tensorf b = Tensorf::LinSpace(-0.1f, 0.2f, 3);

	Symbol* data = NeuralNet::Create<Variable>(x);
	Symbol* weights = NeuralNet::Create<Variable>(w);
	Symbol* biases = NeuralNet::Create<Variable>(b);

	Symbol* convolution = NeuralNet::Create<Convolution>(data, weights, biases, Array<int>({ 4, 4 }), 3, Array<int>({ 2, 2 }), Array<int>({ 1, 1 }));

	NeuralNet net(convolution, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ data, weights, biases });
	symbolsToEvaluate.Add(convolution); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf result = convolution->GetOutput();
	Tensorf correctResult = Tensorf( { { { { -0.08759809f, -0.10987781f },
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
			{2.38090835f, 2.38247847f}}}} );

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(result - correctResult)), 0.0f) << "\n";

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

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(dataGrad - dataNumericalGrad)), 0.0f) << "\n";
	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(weightsGrad - numericalWeightsGrad)), 0.0f) << "\n";
	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(biasesGrad - numericalBiasesGrad)), 0.0f) << "\n";

	NeuralNet::Release();

	return true;
}

bool TestRelu()
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

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(result - correctResult)), 0.0f) << "\n";

	Tensorf dx = symbolsToEvaluate[0]->GetOutput(0);

	Tensorf upperGrads = Tensorf::Ones(result.Shape());
	Tensorf dxNumerical = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ relu });
		return relu->GetOutput();
	},
	x, upperGrads);

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(dx - dxNumerical)), 0.0f) << "\n";

	NeuralNet::Release();

	return true;
}

void TestPooling()
{
	StaticArray<int, 4> xShape = { 2,3,4,4 };
	Tensorf input = Tensorf::LinSpace(-0.3f, 0.4f, Algorithm::Accumulate(xShape, 1, Algorithm::Multiply<>())).Reshape(xShape);

	Symbol* x = NeuralNet::Create<Variable>(input);
	Symbol* pooling = NeuralNet::Create<Pooling>(x, Array<int>({ 2,2 }), Array<int>({ 2,2 }), Array<int>({ 0,0 }));

	NeuralNet net(pooling, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ x });
	symbolsToEvaluate.Add(pooling); // Add loss layer

	net.Execute(symbolsToEvaluate);

	Tensorf result = pooling->GetOutput();
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

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(result - correctResult)), 0.0f) << "\n";

	Tensorf dx = symbolsToEvaluate[0]->GetOutput(0);

	Tensorf upperGrads = Tensorf::Ones(result.Shape());
	Tensorf dxNumerical = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ pooling });
			return pooling->GetOutput();
		},
		x, upperGrads);

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(dx - dxNumerical)), 0.0f) << "\n";

	NeuralNet::Release();
}

//void TestBatchNormalizationForward()
//{
//	{
//		// Check the training-time forward pass by checking means and variances
//		// of features both before and after batch normalization
//		// Simulate the forward pass for a two-layer network
//		const int N = 200, D1 = 50, D2 = 60, D3 = 3;
//		Tensorf X = Tensorf::RandomNormalDistribution(1.0, N, D1);
//		Tensorf W1 = Tensorf::RandomNormalDistribution(1.0, D1, D2);
//		Tensorf W2 = Tensorf::RandomNormalDistribution(1.0, D2, D3);
//		Tensorf a = Tensorf::Dot(Tensorf::ReluActivate(Tensorf::Dot(X, W1)), W2);
//
//		// Before batch normalization
//		Tensorf mean = Tensorf::Mean(a, { 0 });
//		Tensorf stds = Tensorf::StandardDeviation(a, { 0 });
//
//		UniquePtr<Symbol> input = MakeUnique<Variable>(a);
//		UniquePtr<Symbol> scale = MakeUnique<Variable>(Tensorf::Ones(D3));
//		UniquePtr<Symbol> bias = MakeUnique<Variable>(Tensorf::Zeroes(D3));
//
//		UniquePtr<Symbol> BatchNorm = MakeUnique<BatchNormalization>(input.Get(), scale.Get(), bias.Get(), true, 0.9f);
//		BatchNorm->Forward();
//		Tensorf& normalized = BatchNorm->GetOutput();
//
//		// After batch normalization (scale=1, bias=0)
//		// Means should be close to zero and stds close to one
//		mean = Tensorf::Mean(normalized, { 0 });
//		stds = Tensorf::StandardDeviation(normalized, { 0 });
//	}
//
//	{
//		const int N = 200, D1 = 50, D2 = 60, D3 = 3;
//		Tensorf X = Tensorf::RandomNormalDistribution(1.0, N, D1);
//		Tensorf W1 = Tensorf::RandomNormalDistribution(1.0, D1, D2);
//		Tensorf W2 = Tensorf::RandomNormalDistribution(1.0, D2, D3);
//		Tensorf a = Tensorf::Dot(Tensorf::ReluActivate(Tensorf::Dot(X, W1)), W2);
//
//		// Before batch normalization
//		Tensorf mean = Tensorf::Mean(a, { 0 });
//		Tensorf stds = Tensorf::StandardDeviation(a, { 0 });
//
//		UniquePtr<Symbol> input = MakeUnique<Variable>(a);
//		UniquePtr<Symbol> scale = MakeUnique<Variable>(Tensorf({ 1,2,3 }));
//		UniquePtr<Symbol> bias = MakeUnique<Variable>(Tensorf({ 11,12,13 }));
//
//		UniquePtr<Symbol> BatchNorm = MakeUnique<BatchNormalization>(input.Get(), scale.Get(), bias.Get(), true, 0.9f);
//		BatchNorm->Forward();
//		Tensorf& normalized = BatchNorm->GetOutput();
//
//		// After batch normalization
//		// Now means should be close to bias and stds close to scale
//		mean = Tensorf::Mean(normalized, { 0 });
//		stds = Tensorf::StandardDeviation(normalized, { 0 });
//	}
//
//	{
//		// Check the training-time forward pass by checking means and variances
//		// of features both before and after batch normalization
//		// Simulate the forward pass for a two-layer network
//		const int N = 200, D1 = 50, D2 = 60, D3 = 3;
//		Tensorf W1 = Tensorf::RandomNormalDistribution(1.0, D1, D2);
//		Tensorf W2 = Tensorf::RandomNormalDistribution(1.0, D2, D3);
//
//		UniquePtr<Symbol> scale = MakeUnique<Constant>(Tensorf::Ones(D3));
//		UniquePtr<Symbol> bias = MakeUnique<Constant>(Tensorf::Zeroes(D3));
//
//		UniquePtr<Variable> input = MakeUnique<Variable>();
//		UniquePtr<BatchNormalization> BatchNorm = MakeUnique<BatchNormalization>(input.Get(), scale.Get(), bias.Get(), true, 0.9f);
//
//		Tensorf X, a;
//		for (int i = 0; i < 50; i++)
//		{
//			X = Tensorf::RandomNormalDistribution(1.0, N, D1);
//			a = Tensorf::Dot(Tensorf::ReluActivate(Tensorf::Dot(X, W1)), W2);
//
//			input->SetData(a);
//			BatchNorm->Forward();
//		}
//
//		X = Tensorf::RandomNormalDistribution(1.0, N, D1);
//		a = Tensorf::Dot(Tensorf::ReluActivate(Tensorf::Dot(X, W1)), W2);
//
//		input->SetData(a);
//		BatchNorm->mTraining = false;
//
//		BatchNorm->Forward();
//
//		// Before batch normalization
//		Tensorf mean = Tensorf::Mean(a, { 0 });
//		Tensorf stds = Tensorf::StandardDeviation(a, { 0 });
//
//		Tensorf& normalized = BatchNorm->GetOutput();
//
//		// After batch normalization (scale=1, bias=0)
//		// Means should be close to zero and stds close to one
//		mean = Tensorf::Mean(normalized, { 0 });
//		stds = Tensorf::StandardDeviation(normalized, { 0 });
//	}
//
//}

void TestSoftmax()
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

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(result - correctResult)), 0.0f) << "\n";

	Tensorf dx = symbolsToEvaluate[0]->GetOutput(0);

	Tensorf upperGrads = Tensorf::Ones(result.Shape());
	Tensorf dxNumerical = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ softmax });
			return softmax->GetOutput();
		},
		x, upperGrads);

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(dx - dxNumerical)), 0.0f) << "\n";

	NeuralNet::Release();
}

//bool TestDropoutForward()
//{
//	Tensorf dataX = Tensorf::RandomNormalDistribution(1.0, 500, 500) + Scalar(10.0f);
//	float mean = Tensorf::Mean(dataX)[0];
//
//	UniquePtr<Symbol> x = MakeUnique<Variable>(dataX);
//
//	float dropoutProb[3] = { 0.3f, 0.6f, 0.75f };
//
//	for (int i = 0; i < 3; i++)
//	{
//		{
//			UniquePtr<Dropout> dropout = MakeUnique<Dropout>(x.Get(), dropoutProb[i], true);
//			dropout->mTraining = true;
//			dropout->Forward();
//			Tensorf trainingOut = dropout->GetOutput();
//			if (Math::Abs(Tensorf(Tensorf::Mean(trainingOut) - Scalar(10) * (Scalar(1.0f) - Scalar(dropoutProb[i])))[0]) > 0.02f)
//				return false;
//
//			dropout->mTraining = false;
//			dropout->Forward();
//			Tensorf inferenceOut = dropout->GetOutput();
//
//			if (Math::Abs(Tensorf(Tensorf::Mean(inferenceOut) - Scalar(10.0f))[0]) > 0.02f)
//				return false;
//		}
//	}
//
//	return true;
//}

void TestAdam()
{
	int N = 4, D = 5;
	Tensorf w = Tensorf::LinSpace(-0.4f, 0.6f, N * D).Reshape(N, D);
	Tensorf dw = Tensorf::LinSpace(-0.6f, 0.4f, N * D).Reshape(N, D);
	Tensorf m = Tensorf::LinSpace(0.6f, 0.9f, N * D).Reshape(N, D);
	Tensorf v = Tensorf::LinSpace(0.7f, 0.5f, N * D).Reshape(N, D);

	Adam adam(1e-2f, 0.9f, 0.999f, 1e-8f, 5);
	adam.mMapM.Add(&w, m);
	adam.mMapV.Add(&w, v);

	adam.Step(w, dw);

	Tensorf expected_next_w = {
		{-0.40094747f, -0.34836187f, -0.29577703f, -0.24319299f, -0.19060977f},
		{-0.1380274f, -0.08544591f, -0.03286534f, 0.01971428f, 0.0722929f},
		{0.1248705f, 0.17744702f, 0.23002243f, 0.28259667f, 0.33516969f},
		{0.38774145f, 0.44031188f, 0.49288093f, 0.54544852f, 0.59801459f} };
	Tensorf expected_v = {
		{0.69966f, 0.68908382f, 0.67851319f, 0.66794809f, 0.65738853f, },
		{0.64683452f, 0.63628604f, 0.6257431f, 0.61520571f, 0.60467385f, },
		{0.59414753f, 0.58362676f, 0.57311152f, 0.56260183f, 0.55209767f, },
		{0.54159906f, 0.53110598f, 0.52061845f, 0.51013645f, 0.49966f, } };
	Tensorf expected_m = {
		{0.48f, 0.49947368f, 0.51894737f, 0.53842105f, 0.55789474f},
		{0.57736842f, 0.59684211f, 0.61631579f, 0.63578947f, 0.65526316f},
		{0.67473684f, 0.69421053f, 0.71368421f, 0.73315789f, 0.75263158f},
		{0.77210526f, 0.79157895f, 0.81105263f, 0.83052632f, 0.85f} };

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(w - expected_next_w)), 0.0f) << "\n";
	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(adam.mMapV[&w] - expected_v)), 0.0f) << "\n";
	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(adam.mMapM[&w] - expected_m)), 0.0f) << "\n";
}

void TestContentLoss()
{
	Tensorf imageFeatureTensor = Tensorf::RandomNormalDistribution(256.0f, 1, 3, 4, 4);
	Tensorf contentFeatureTensor = Tensorf::RandomNormalDistribution(128.0f, 1, 3, 4, 4);

	Symbol* imageFeature = NeuralNet::Create<Variable>(imageFeatureTensor);
	Symbol* contentFeature = NeuralNet::Create<Constant>(contentFeatureTensor);

	Symbol* contentLoss = NeuralNet::Create<ContentLoss>(imageFeature, contentFeature);

	NeuralNet net(contentLoss, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ imageFeature });
	symbolsToEvaluate.Add(contentLoss); // Add loss layer

	net.Execute(symbolsToEvaluate);

	std::cout << contentLoss->GetOutput()[0] << "\n";

	Tensorf imageGrad = symbolsToEvaluate[0]->GetOutput(imageFeature->GetGradientIndex());

	Tensorf upperGrads = Tensorf::Ones(1);
	Tensorf imageNumericalGrad = NumericalGradientEval([&]() -> Tensorf
	{
		net.Execute({ contentLoss });
		return contentLoss->GetOutput();
	},
		imageFeature, upperGrads, 0.1f);

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(imageGrad - imageNumericalGrad)), 0.0f) << "\n";

	NeuralNet::Release();
}

void TestStyleLoss()
{
	Tensorf imageFeatureTensor = Tensorf::RandomNormalDistribution(256.0f, 1, 3, 4, 4);
	Tensorf styleFeatureTensor = Tensorf::RandomNormalDistribution(128.0f, 1, 3, 4, 4);

	Symbol* imageFeature = NeuralNet::Create<Variable>(imageFeatureTensor);
	Symbol* styleFeature = NeuralNet::Create<Constant>(styleFeatureTensor);

	Symbol* styleLoss = NeuralNet::Create<StyleLoss>(imageFeature, styleFeature);

	NeuralNet net(styleLoss, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ imageFeature });
	symbolsToEvaluate.Add(styleLoss); // Add loss layer

	net.Execute(symbolsToEvaluate);

	std::cout << styleLoss->GetOutput()[0] << "\n";

	Tensorf imageGrad = symbolsToEvaluate[0]->GetOutput(imageFeature->GetGradientIndex());

	Tensorf upperGrads = Tensorf::Ones(1);
	Tensorf imageNumericalGrad = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ styleLoss });
			return styleLoss->GetOutput();
		},
		imageFeature, upperGrads, 0.1f);

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(imageGrad - imageNumericalGrad)), 0.0f) << "\n";

	NeuralNet::Release();
}

void TestTotalVariationLoss()
{
	Tensorf imageTensor = Tensorf::RandomFloat(1, 3, 6, 6);

	Symbol* image = NeuralNet::Create<Variable>(imageTensor);
	Symbol* tvLoss = NeuralNet::Create<TotalVariationLoss>(image);

	NeuralNet net(tvLoss, true);

	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols({ image });
	symbolsToEvaluate.Add(tvLoss); // Add loss layer

	net.Execute(symbolsToEvaluate);

	std::cout << tvLoss->GetOutput()[0] << "\n";

	Tensorf imageGrad = symbolsToEvaluate[0]->GetOutput(image->GetGradientIndex());

	Tensorf upperGrads = Tensorf::Ones(1);
	Tensorf imageNumericalGrad = NumericalGradientEval([&]() -> Tensorf
		{
			net.Execute({ tvLoss });
			return tvLoss->GetOutput();
		},
		image, upperGrads, 0.001f);

	std::cout << Algorithm::Accumulate(Tensorf(Tensorf::Abs(imageGrad - imageNumericalGrad)), 0.0f) << "\n";

	NeuralNet::Release();
}

void TestMattingLaplacian()
{
	Tensorf A = Tensorf({
		{ { 0.79321955f,  0.26368653f,  0.06885819f },
		{ 0.24796998f,  0.7146999f,  0.35786379f },
		{ 0.34516456f,  0.27023826f,  0.18541087f },
		{ 0.74764321f,  0.65248568f,  0.83015837f } },

		{ { 0.30999182f,  0.05887338f,  0.26157566f },
		{ 0.96582913f, 0.62514031f, 0.15679928f },
		{ 0.11762355f, 0.63102276f, 0.92336963f },
		{ 0.57942699f, 0.75011694f, 0.12483227f } },

		{ { 0.43055699f,  0.60420677f,  0.0049846f },
		{ 0.37634474f, 0.87662164f, 0.46829144f },
		{ 0.53187853f, 0.66366641f, 0.04956319f },
		{ 0.68627279f, 0.89384646f, 0.78752079f } },

		{ { 0.16140868f,  0.46934196f,  0.76709362f },
		{ 0.03980935f, 0.48688648f, 0.30714676f },
		{ 0.68860603f, 0.4019053f, 0.91339134f },
		{ 0.26836705f, 0.55345901f, 0.13667431f } }
	});

	A = Tensorf::Transpose(A, { 2, 0, 1 });
	SparseMatrixf laplacian = MattingLaplacian::CalcLaplacianMatrix(A);
}

void main()
{
	TestFullyConnected();
	TestRelu();
	TestPooling();
	TestConvolution();
	TestSoftmax();
	//TestBatchNormalizationForward();
	//TestDropoutForward();
	TestAdam();
	TestStyleLoss();
	TestContentLoss();
	TestTotalVariationLoss();

	//TestMattingLaplacian();

	Assertf(!_CrtDumpMemoryLeaks(), "Memory leak detected. See debug output for details");
}