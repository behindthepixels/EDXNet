#include "Relu.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Relu::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<ReluGradientSymbol>(mInputs[0], pUpperGrads);
		}
	}
}