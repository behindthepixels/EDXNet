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

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const Tensorf& labels = mInputs[1]->GetOutput();

				Tensorf probs = Tensorf::Exp(inputValue - Tensorf::Max(inputValue, { 1 }, true));
				probs /= Tensorf::Sum(probs, { 1 }, true);

				const int N = inputValue.Shape(0);

				Tensorf correctProbs;
				correctProbs.Resize(N);
				for (int i = 0; i < N; i++)
				{
					correctProbs[i] = probs(i, int(labels[i]));
				}

				// Loss
				Tensorf& output = GetOutput(0);
				output = Scalar(-1.0f) * Tensorf::Sum(Tensorf::Log(correctProbs)) / Scalar(N);
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
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

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const Tensorf& labels = mInputs[1]->GetOutput();

				Tensorf probs = Tensorf::Exp(inputValue - Tensorf::Max(inputValue, { 1 }, true));
				probs /= Tensorf::Sum(probs, { 1 }, true);

				const int N = inputValue.Shape(0);
				for (int i = 0; i < N; i++)
					probs(i, int(labels[i])) -= 1.0f;

				// dx
				Tensorf& output = GetOutput(0);
				output = probs / Scalar(float(N));
			}
		};
	}
}
