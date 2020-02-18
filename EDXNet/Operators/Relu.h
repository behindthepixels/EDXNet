#pragma once


#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Relu : public SymbolBase<1, 1>
		{
		public:
			Relu(Symbol* pInput)
			{
				mInputs[0] = pInput;
			}

			void Evaluate() override;

			void Init()
			{
				const auto& inputShape = mInputs[0]->GetOutput().Shape();

				Tensorf& output = GetOutput();
				output.Resize(inputShape);
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;
		};

		class ReluGradientSymbol : public SymbolBase<2, 1>
		{
		public:
			ReluGradientSymbol(Symbol* pInput, Symbol* pUpperGradients)
			{
				mInputs[0] = pInput;
				mInputs[1] = pUpperGradients;
			}

			void Evaluate() override;
		};
	}
}