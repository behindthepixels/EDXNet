#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Softmax : public SymbolBase<2, 1>
		{
		public:
			Softmax(Symbol* pInputVal, Symbol* pLabels)
			{
				mInputs[0] = pInputVal;
				mInputs[1] = pLabels;
			}

			void Evaluate() override;

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			Tensorf mProbs;
			Tensorf mCorrectProbs;
			float mRegularizationStrength = 1e-5f;
		};

		class SoftmaxGradient : public SymbolBase<2, 1>
		{
		public:
			SoftmaxGradient(Symbol* pInputVal, Symbol* pLabels)
			{
				mInputs[0] = pInputVal;
				mInputs[1] = pLabels;
			}

			void Evaluate() override;

		private:
			Tensorf mProbs;
		};
	}
}
