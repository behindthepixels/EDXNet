#pragma once

#include "../Core/NeuralNet.h"
#include "VGG19.h"

namespace EDX
{
	namespace DeepLearning
	{
		class StyleTransferVGG19
		{
		public:
			VGG19 contentNet;
			VGG19 styleNet;
			VGG19 transferNet;

			Symbol* contentLoss;
			Symbol* styleLoss1;
			Symbol* styleLoss2;
			Symbol* styleLoss3;
			Symbol* styleLoss4;
			Symbol* styleLoss5;

			Symbol* totalVariationLoss;

			Symbol* styleLossSum;

			Symbol* transferLoss;

		public:
			static StyleTransferVGG19 Create(const TCHAR* weightsPath,
				const float contentWeight = 5.0f,
				const float styleWeight = 400.0f,
				const float totalVariationWeight = 1e-3f);
			static StyleTransferVGG19 Create(const VGG19Weights& weights,
				const float contentWeight = 5.0f,
				const float styleWeight = 400.0f,
				const float totalVariationWeight = 1e-3f);
		};
	}
}