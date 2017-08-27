#include "BatchNormalization.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* BatchNormalization::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<BatchNormalizationGradient>(
				mInputs[0],
				mInputs[1],
				mInputs[2],
				(Symbol*)(this),
				pUpperGrads);
		}
	}
}