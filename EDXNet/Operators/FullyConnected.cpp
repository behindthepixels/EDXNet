#include "FullyConnected.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* FullyConnected::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<FullyConnectedGradient>(pUpperGrads, mInputs[0], mInputs[1], mInputs[2]);
		}
	}
}