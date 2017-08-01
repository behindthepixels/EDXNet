#include "Convolution.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Convolution::Gradient(Symbol* pUpperGrads) const
		{
			return NeuralNet::Create<ConvolutionGradient>(
				mInputs[0],
				mInputs[1],
				mInputs[2],
				(Symbol*)(this),
				pUpperGrads,
				mKernelSize,
				mNumFilter,
				mStride,
				mPadding);
		}
	}
}