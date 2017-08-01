
#include "Core/EDXNet.h"
#include "Models/LeNet.h"
#include "Models/VGG19.h"

#include "Windows/Bitmap.h"
#include "Graphics/Color.h"
#include "Graphics/Texture.h"

using namespace EDX;
using namespace EDX::DeepLearning;
using namespace EDX::Algorithm;


Color* ImageResample(const Texture2D<Color>* pTex, const int width, const int height)
{
	Color* pRet = new Color[width * height];

	Vector2 deriv[2] = {
		Vector2(1 / float(width), 0.0f),
		Vector2(0.0f, 1 / float(height)),
	};
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			pRet[i * width + j] = pTex->Sample(Vector2(j / float(width), i / float(height)),
				deriv,
				TextureFilter::Anisotropic8x);
		}
	}

	return pRet;
}

void main()
{
	int imgWidth, imgHeight, imgChannel;
	uint8* pImage = (uint8*)Bitmap::ReadFromFile<Color4b>("coffee.jpg", &imgWidth, &imgHeight, &imgChannel);

	ImageTexture<Color, Color4b> tex((Color4b*)pImage, imgWidth, imgHeight);
	Color* resizedTexture = ImageResample(&tex, 224, 224);

	float* pfImage = new float[224 * 224 * 3];
	for (int i = 0; i < 224 * 224; i++)
	{
		// Swap R and B channel
		pfImage[i * 3 + 0] = resizedTexture[i].b * 255.0f;
		pfImage[i * 3 + 1] = resizedTexture[i].g * 255.0f;
		pfImage[i * 3 + 2] = resizedTexture[i].r * 255.0f;
	}

	Tensorf image;
	image.Assign(pfImage, { 1, 224, 224, 3 });
	image -= DataSet::ImageNet::GetMeanImage();
	image = Tensorf::Transpose(image, { 0, 3, 1, 2 });

	delete[] pImage;
	delete[] resizedTexture;
	delete[] pfImage;

	VGG19 vgg19 = VGG19::CreateForInference("../Models/VGG19.dat");

	vgg19.input->SetData(image);
	NeuralNet net(vgg19.loss, false, false);

	net.Execute({ vgg19.loss });
	std::cout << DataSet::ImageNet::GetLabelString()[vgg19.loss->GetOutput()(0, 0)] << std::endl;

	NeuralNet::Release();
}
