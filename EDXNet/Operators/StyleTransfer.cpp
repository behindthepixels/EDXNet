#include "StyleTransfer.h"

namespace EDX
{
	namespace DeepLearning
	{

		Symbol* ContentLoss::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<ContentLossGradient>((Symbol*)(this), mInputs[0], mInputs[1], pUpperGrads);
		}

		Symbol* StyleLoss::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<StyleLossGradient>((Symbol*)(this), mInputs[0], mInputs[1], pUpperGrads);
		}

		Symbol* TotalVariationLoss::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<TotalVariationGradient>((Symbol*)(this), mInputs[0], pUpperGrads);
		}

		Symbol* PhotorealismLoss::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<PhotorealismGradient>(mInputs[0], pUpperGrads, mLaplacianMatrix);
		}

		Symbol* StyleTransferLoss::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<StyleTransferLossGradient>(mInputs[0], mInputs[1], mInputs[2], pUpperGrads, mContentWeight, mStyleWeight, mTotalVariartionWeight);
		}
	}
}