#include "Dropout.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Dropout::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<DropoutGradient>(
				(Symbol*)(this),
				pUpperGrads,
				mTraining);
		}
	}
}