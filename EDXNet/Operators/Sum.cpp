#include "Sum.h"

namespace EDX
{
	namespace DeepLearning
	{
		Symbol* Sum::Gradient(Symbol* pUpperGrads) const
		{
			Array<Symbol*> pGradInput;
			for (int i = 0; i < mNumInputs; i++)
				pGradInput.Add(mInputs[i]);

			return NeuralNet::Create<SumGradient>(pGradInput, (Symbol*)(this), pUpperGrads);
		}
	}
}