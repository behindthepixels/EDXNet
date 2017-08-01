#include "Relu.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Relu::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<ReluGradient>(mInputs[0], pUpperGrads);
		}
	}
}