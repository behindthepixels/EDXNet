#include "Pooling.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Pooling<MaxPoolOp>::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<PoolingGradient<MaxPoolGradOp>>(mInputs[0], (Symbol*)(this), pUpperGrads, mKernelSize, mStride, mPadding);
		}

		Symbol* Pooling<AvgPoolOp>::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<PoolingGradient<AvgPoolGradOp>>(mInputs[0], (Symbol*)(this), pUpperGrads, mKernelSize, mStride, mPadding);
		}
	}
}