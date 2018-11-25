#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Predictor : public SymbolBase<1, 1>
		{
		public:
			Predictor(Symbol* pInputVal)
			{
				mInputs[0] = pInputVal;
			}

			void Evaluate() override;
		};
	}
}