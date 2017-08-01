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

			void Evaluate() override
			{
				Tensorf& inputValue = mInputs[0]->GetOutput();
				const Tensorf& weights = mInputs[1]->GetOutput();
				const Tensorf& bias = mInputs[2]->GetOutput();

				const int N = inputValue.Shape(0);
				const int dim = inputValue.LinearSize() / N;
				Tensorf X = inputValue.GetWithShape(N, dim);

				Assertf(!X.Empty(), "Input data cannot be empty in fully connected layer.");

				GetOutput() = Tensorf::Dot(X, weights) + bias;
			}

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

			void Evaluate() override
			{
				Tensorf& inputValue = mInputs[0]->GetOutput();
				Tensorf& weights = mInputs[1]->GetOutput();
				const Tensorf& bias = mInputs[2]->GetOutput();
				const Tensorf& upperGrads = mInputs[3]->GetOutput();

				Tensorf& inputGrads = GetOutput(0);
				Tensorf& weightsGrads = GetOutput(1);
				Tensorf& biasesGrads = GetOutput(2);

				const int N = inputValue.Shape(0);
				const int dim = inputValue.LinearSize() / inputValue.Shape(0);
				Tensorf X = inputValue.GetWithShape(N, dim);

				inputGrads = Tensorf::Dot(upperGrads, weights.GetTransposed());
				weightsGrads = Tensorf::Dot(X.GetTransposed(), upperGrads);
				biasesGrads = Tensorf::Sum(upperGrads, { 0 });

				mInputs[0]->SetGradientIndex(0);
				mInputs[1]->SetGradientIndex(1);
				mInputs[2]->SetGradientIndex(2);
			}
		};
	}
}