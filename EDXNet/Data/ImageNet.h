#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		namespace DataSet
		{
			class ImageNet
			{
			public:
				static StaticArray<TCHAR*, 1000> GetLabelString();
				static Tensor<float, CPU> GetMeanImage()
				{
					return{ 103.939f , 116.779f , 123.68f };
				}
			};
		}
	}
}