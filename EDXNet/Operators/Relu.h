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

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();

				GetOutput() = Tensorf::ReluActivate(inputValue);
			}

			void Init()
			{
				const auto& inputShape = mInputs[0]->GetOutput().Shape();

				Tensorf& output = GetOutput();
				output.Resize(inputShape);
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;
		};

		class ReluGradient : public SymbolBase<2, 1>
		{
		public:
			ReluGradient(Symbol* pInput, Symbol* pUpperGradients)
			{
				mInputs[0] = pInput;
				mInputs[1] = pUpperGradients;
			}

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const Tensorf& upperGrads = mInputs[1]->GetOutput();

				Tensorf& output = GetOutput();
				output = upperGrads;

				for (int i = 0; i < output.LinearSize(); i++)
				{
					if (inputValue[i] < 0.0f)
						output[i] = 0.0f;
				}

				mInputs[0]->SetGradientIndex(0);
			}
		};
	}
}