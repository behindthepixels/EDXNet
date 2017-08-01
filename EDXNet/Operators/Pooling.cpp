#include "Pooling.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Pooling::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<PoolingGradient>(mInputs[0], (Symbol*)(this), pUpperGrads, mKernelSize, mStride, mPadding);
		}
	}
}