#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class FullyConnected : public SymbolBase<3, 1>
		{
		public:
			FullyConnected(Symbol* pInputValue, Symbol* pWeights, Symbol* pBias, const int numHidden)
			{
				mInputs[0] = pInputValue;
				mInputs[1] = pWeights;
				mInputs[2] = pBias;

				mNumHidden = numHidden;
			}

			void Init() override;
			void Evaluate() override;
			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			int mNumHidden;
		};

		class FullyConnectedGradient : public SymbolBase<4, 3>
		{
		public:
			FullyConnectedGradient(Symbol* pUpperGradients, Symbol* pInputValue, Symbol* pWeights, Symbol* pBias)
			{
				mInputs[0] = pInputValue;
				mInputs[1] = pWeights;
				mInputs[2] = pBias;
				mInputs[3] = pUpperGradients;
			}

			void Evaluate() override;
		};
	}
}