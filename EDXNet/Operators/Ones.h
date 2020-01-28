#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Ones : public SymbolBase<1, 1>
		{
		public:
			Ones(Symbol* pInputVal)
			{
				mInputs[0] = pInputVal;
			}

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();

				Tensorf& ret = GetOutput();
				ret = TensorExpr::Ones(inputValue.Shape());
			}

			void Init()
			{
				const auto& inputShape = mInputs[0]->GetOutput().Shape();

				Tensorf& output = GetOutput();
				output.Resize(inputShape);
			}

		private:
			Array<int> mShape;
		};
	}
}