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

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();

				// Predicted result
				const int N = inputValue.Shape(0);
				Tensorf& output = GetOutput(0);
				output.Resize(N);

				for (int i = 0; i < N; i++)
				{
					float max = Math::EDX_NEG_INFINITY;
					int index = -1;
					for (int j = 0; j < inputValue.Shape(1); j++)
					{
						float currentVal = inputValue(i, j);
						if (currentVal > max)
						{
							max = currentVal;
							index = j;
						}
					}

					output[i] = index;
				}
			}
		};
	}
}