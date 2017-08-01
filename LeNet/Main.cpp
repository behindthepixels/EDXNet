
#include "Core/EDXNet.h"
#include "Models/LeNet.h"

#include "Core/SmartPointer.h"
#include "Windows/Bitmap.h"
#include "Graphics/Color.h"
#include "Graphics/Texture.h"
#include "Windows/stb_image.h"
#include "Windows/Threading.h"
#include "Windows/Timer.h"

using namespace EDX;
using namespace EDX::DeepLearning;
using namespace EDX::Algorithm;

void LeNetRun()
{
	DataSet::MNIST mnist;
	mnist.Load("D:/Coding/DataSet/MNIST");

	Tensorf& dataTensor = mnist.GetTestData();
	Tensorf& labelTensor = mnist.GetTestLabels();

	LeNet leNet = LeNet::CreateForInference("../Models/LeNet.dat");
	leNet.input = NeuralNet::Create<Constant>(dataTensor);

	NeuralNet net(leNet.leNet);
	net.Execute({ leNet.leNet });

	Tensorf result = leNet.leNet->GetOutput();

	int N = labelTensor.Shape(0);
	int correct = 0;
	for (int i = 0; i < labelTensor.LinearSize(); i++)
	{
		if (labelTensor[i] == result[i])
			correct++;
	}

	std::cout << "Accuracy: " << correct / float(N) << "\n";

	NeuralNet::Release();
}

void LeNetTrain()
{
	DataSet::MNIST mnist;
	mnist.Load("D:/Coding/DataSet/MNIST");

	Tensorf& dataTensor = mnist.GetTrainingData();
	Tensorf& labelTensor = mnist.GetTrainingLabels();

	LeNet leNet = LeNet::CreateForTraining(dataTensor, labelTensor);

	NeuralNet net(leNet.leNet, true, true);

	Array<Symbol*> symbolsToTrain = net.GetVariableSymbols();
	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols(symbolsToTrain);
	symbolsToEvaluate.Add(leNet.leNet); // Add loss layer

	const Array<int>& dataShape = dataTensor.Shape();

	Adam adam;

	float mLearningRate = 0.01f;
	int mMaxEpoches = 1000;
	int mMinibatchSize = 128;
	for (int epochIter = 0; epochIter < mMaxEpoches; epochIter++)
	{
		adam.SetLearningRate(mLearningRate);

		int N = dataTensor.Shape(0);
		std::cout << "Epoch # " << epochIter << ".\n";
		for (int epochPos = 0; epochPos < N; epochPos += mMinibatchSize)
		{
			Tensorf inputSlice = dataTensor.GetSection(epochPos, Math::Min(N, epochPos + mMinibatchSize));
			Tensorf labelSlice = labelTensor.GetSection(epochPos, Math::Min(N, epochPos + mMinibatchSize));

			leNet.input->SetData(inputSlice);
			leNet.labels->SetData(labelSlice);

			net.Execute(symbolsToEvaluate);

			for (int i = 0; i < symbolsToTrain.Size(); i++)
			{
				Symbol* symbol = symbolsToTrain[i];
				Tensorf& param = symbol->GetOutput();

				const Tensorf& gradient = symbolsToEvaluate[i]->GetOutput(symbol->GetGradientIndex());
				adam.Step(param, gradient);
			}

			std::cout << "Loss: " << leNet.leNet->GetOutput()[0] << "\n";
		}

		mLearningRate *= 0.995f;

		{
			FileStream stream("LeNet_Trained.dat", FileMode::Create);
			stream << leNet;
			stream.Close();
		}
	}

	NeuralNet::Release();
}

bool bTraining = true;

void main()
{
	if (bTraining)
		LeNetTrain();
	else
		LeNetRun();
}