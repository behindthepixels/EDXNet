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

			void Init() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();

				const int N = inputValue.Shape(0);
				const int dim = inputValue.LinearSize() / inputValue.Shape(0);

				Array<int> weightsShape = { dim, mNumHidden };

				Tensorf& weights = mInputs[1]->GetOutput();
				Tensorf& biases = mInputs[2]->GetOutput();

				weights = Tensorf::RandomNormalDistribution(0.1f, dim, mNumHidden);
				biases = Tensorf::Zeroes(mNumHidden);

				Tensorf& output = GetOutput();
				output.Resize(N, mNumHidden);
			}

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