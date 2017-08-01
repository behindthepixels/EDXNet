#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Dropout : public SymbolBase<1, 2>
		{
		public:
			Dropout(Symbol* pInput, const float dropProb, const bool training)
				: mDropProb(dropProb)
				, mTraining(training)
			{
				mInputs[0] = pInput;
			}

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				Tensorf& output = GetOutput(0);

				if (mTraining)
				{
					output.Resize(inputValue.Shape());

					Tensorf& mask = GetOutput(1);
					mask.Resize(inputValue.Shape());

					RandomGen random;
					for (int i = 0; i < inputValue.LinearSize(); i++)
					{
						mask[i] = random.Float() < mDropProb ? 0.0f : 1.0f;
						output[i] = inputValue[i] * mask[i];
					}
				}
				else
				{
					output = inputValue;
				}
			}

			void Init()
			{
				const auto& inputShape = mInputs[0]->GetOutput().Shape();

				Tensorf& output = GetOutput();
				output.Resize(inputShape);
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		public:
			float mDropProb;
			bool mTraining;
		};

		class DropoutGradient : public SymbolBase<2, 1>
		{
		public:
			DropoutGradient(Symbol* dropout, Symbol* pUpperGrads, const bool training)
				: mTraining(training)
			{
				mInputs[0] = dropout;
				mInputs[1] = pUpperGrads;
			}

			void Evaluate() override
			{
				const Tensorf& upperGrads = mInputs[1]->GetOutput();
				Tensorf& output = GetOutput(0);

				if (mTraining)
				{
					const Tensorf& mask = mInputs[0]->GetOutput(1);
					output = upperGrads * mask;
				}
				else
				{
					output = upperGrads;
				}
			}

		public:
			bool mTraining;

		};
	}
}