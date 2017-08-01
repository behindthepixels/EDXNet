
#include "Core/EDXNet.h"
#include "Models/StyleTransferVGG19.h"

#include "Windows/Timer.h"
#include "Windows/Bitmap.h"

using namespace EDX;
using namespace EDX::DeepLearning;
using namespace EDX::Algorithm;


void main()
{
	// ------------------------------------------------------------------------------
	// Initialize content, style and transferred images
	// ------------------------------------------------------------------------------
	Tensorf contentImageTensor;
	{
		int imgWidth, imgHeight, imgChannel;
		uint8* pImage = Bitmap::ReadFromFile<uint8>("coffee.jpg", &imgWidth, &imgHeight, &imgChannel, 3);
		float* pfImage = new float[imgHeight * imgWidth * 3];

		for (int i = 0; i < imgHeight * imgWidth; i++)
		{
			// Swap R and B channel
			pfImage[i * 3 + 0] = pImage[i * 3 + 2] - DataSet::ImageNet::GetMeanImage()[2];
			pfImage[i * 3 + 1] = pImage[i * 3 + 1] - DataSet::ImageNet::GetMeanImage()[1];
			pfImage[i * 3 + 2] = pImage[i * 3 + 0] - DataSet::ImageNet::GetMeanImage()[0];
		}

		contentImageTensor.Assign(pfImage, { 1, imgHeight, imgWidth, imgChannel });
		contentImageTensor = Tensorf::Transpose(contentImageTensor, { 0, 3, 1, 2 });

		delete[] pImage;
		delete[] pfImage;
	}

	//contentImageTensor = Tensorf::RandomInt(128, 1, 3, 4, 4);

	Tensorf styleImageTensor;
	{
		int imgWidth, imgHeight, imgChannel;
		uint8* pImage = Bitmap::ReadFromFile<uint8>("picasso_selfport1907.jpg", &imgWidth, &imgHeight, &imgChannel, 3);
		float* pfImage = new float[imgHeight * imgWidth * 3];

		for (int i = 0; i < imgHeight * imgWidth; i++)
		{
			// Swap R and B channel
			pfImage[i * 3 + 0] = pImage[i * 3 + 2] - DataSet::ImageNet::GetMeanImage()[2];
			pfImage[i * 3 + 1] = pImage[i * 3 + 1] - DataSet::ImageNet::GetMeanImage()[1];
			pfImage[i * 3 + 2] = pImage[i * 3 + 0] - DataSet::ImageNet::GetMeanImage()[0];
		}

		styleImageTensor.Assign(pfImage, { 1, imgHeight, imgWidth, imgChannel });
		styleImageTensor = Tensorf::Transpose(styleImageTensor, { 0, 3, 1, 2 });

		delete[] pImage;
		delete[] pfImage;
	}

	// Initialize the transferred image as content image by default
	//Tensorf transferredImageTensor = Tensorf::RandomFloat(styleImageTensor.Shape()) * Scalar(40.0f) - Scalar(20.0f);
	Tensorf transferredImageTensor = contentImageTensor;

	// ------------------------------------------------------------------------------
	// Initialize style transfer net
	// ------------------------------------------------------------------------------

	// Default weights
	float contentWeight = 5e0f;
	float styleWeight = 1e4f;
	float tvWeight = 1e-2f;
	StyleTransferVGG19 styleTransfer = StyleTransferVGG19::Create("../Models/VGG19.dat", contentWeight, styleWeight, tvWeight);

	styleTransfer.contentNet.input->SetData(contentImageTensor);
	styleTransfer.styleNet.input->SetData(styleImageTensor);
	styleTransfer.transferNet.input->SetData(transferredImageTensor);


	// ------------------------------------------------------------------------------
	// Optimization
	// ------------------------------------------------------------------------------
	NeuralNet net(styleTransfer.transferLoss, true);

	Array<Symbol*> symbolsToTrain = net.GetVariableSymbols();
	Array<Symbol*> symbolsToEvaluate = net.GetGradientSymbols(symbolsToTrain);
	symbolsToEvaluate.Add(styleTransfer.transferLoss);

	Timer timer;
	timer.Start();

	struct LBFGSInstance
	{
		NeuralNet* pNet;
		Symbol* pTransferredImage;
		Array<Symbol*>* pSymbolsToEval;
		Array<Symbol*>* pSymbolsToTrain;
		Timer* pTimer;
	};

	LBFGSInstance instance = {
		&net,
		styleTransfer.transferNet.input,
		&symbolsToEvaluate,
		&symbolsToTrain,
		&timer
	};

	auto Eval = [](
		void *instance,
		const int n,
		const float *x,
		float *g,
		const float step
		)
	{
		LBFGSInstance* netInstance = (LBFGSInstance*)instance;

		Tensorf& data = netInstance->pTransferredImage->GetOutput();
		data.Assign(x, data.Shape());

		netInstance->pNet->Execute(*(netInstance->pSymbolsToEval));
		float loss = netInstance->pSymbolsToEval->Last()->GetOutput()[0];

		Symbol* symbol = (*(netInstance->pSymbolsToTrain))[0];
		Tensorf& param = symbol->GetOutput();

		Tensorf& gradient = (*(netInstance->pSymbolsToEval))[0]->GetOutput(symbol->GetGradientIndex());
		for (int i = 0; i < gradient.LinearSize(); i++)
			g[i] = gradient[i];

		return loss;
	};

	auto Progress = [](
		void *instance,
		int n,
		const float *x,
		const float *g,
		const float fx,
		const float xnorm,
		const float gnorm,
		const float step,
		int k,
		int ls
		)
	{
		LBFGSInstance* netInstance = (LBFGSInstance*)instance;

		std::cout << "Iteration: " << k << ", ";
		std::cout << "Loss: " << fx << ", ";
		std::cout << "Elapsed " << netInstance->pTimer->GetElapsedTime() << " seconds.\n";

		Tensorf currentImage = netInstance->pTransferredImage->GetOutput();
		currentImage = Tensorf::Transpose(currentImage, { 0, 2, 3, 1 });

		float* imageData = currentImage.Data();

		uint8* imageDataByte = new uint8[currentImage.Shape(1) * currentImage.Shape(2) * 4];

		for (int i = 0; i < currentImage.Shape(1); i++)
		{
			for (int j = 0; j < currentImage.Shape(2); j++)
			{
				// Swap R and B channel
				int imageWriteIdx = (i * currentImage.Shape(2) + j);
				int imageReadIdx = ((currentImage.Shape(1) - i - 1) * currentImage.Shape(2) + j);
				imageDataByte[imageWriteIdx * 4 + 0] = Math::Clamp(imageData[imageReadIdx * 3 + 2] + DataSet::ImageNet::GetMeanImage()[0], 0, 255);
				imageDataByte[imageWriteIdx * 4 + 1] = Math::Clamp(imageData[imageReadIdx * 3 + 1] + DataSet::ImageNet::GetMeanImage()[1], 0, 255);
				imageDataByte[imageWriteIdx * 4 + 2] = Math::Clamp(imageData[imageReadIdx * 3 + 0] + DataSet::ImageNet::GetMeanImage()[2], 0, 255);
				imageDataByte[imageWriteIdx * 4 + 3] = 255;
			}
		}


		Bitmap::SaveBitmapFile("Test.bmp", imageDataByte, currentImage.Shape(2), currentImage.Shape(1));
		delete[] imageDataByte;

		return 0;
	};

	float loss;
	Tensorf data = styleTransfer.transferNet.input->GetOutput();
	lbfgs(data.LinearSize(), data.Data(), &loss, Eval, Progress, &instance, nullptr);

	NeuralNet::Release();
}