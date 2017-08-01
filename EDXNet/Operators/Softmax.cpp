#include "Softmax.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Softmax::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<SoftmaxGradient>(mInputs[0], mInputs[1]);
		}
	}
}