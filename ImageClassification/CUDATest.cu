
#include "Core/EDXNet.h"
#include "Models/VGG19.h"

using namespace EDX;
using namespace EDX::DeepLearning;
using namespace EDX::Algorithm;

void VGG19CUDA(const float* pfImage)
{
	cublasStatus_t status;
	status = cublasCreate(&Cublas::GetHandle());

	// Todo: Dummy way to force the compiler to compile the += opeartor of Tensor class with nvcc
	{
		Tensorf A = { 0 };
		Tensorf B = { 0 };

		A += B;
	}

	Tensor<float, CPU> image;
	image.Assign(pfImage, { 1, 224, 224, 3 });
	image -= DataSet::ImageNet::GetMeanImage();
	image = Tensor<float, CPU>::Transpose(image, { 0, 3, 1, 2 });

	VGG19 vgg19 = VGG19::CreateForInference("../Models/VGG19.dat");

	Tensorf imageGPU = image;
	vgg19.input->SetData(imageGPU);
	NeuralNet net(vgg19.loss, false, false);

	net.Execute({ vgg19.loss });

	Tensor<float, CPU> index = vgg19.loss->GetOutput();
	std::cout << DataSet::ImageNet::GetLabelString()[index(0, 0)] << std::endl;

	NeuralNet::Release();
}