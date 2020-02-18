#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		static const int MAX_INPUT = 8;
		class SumSymbol : public SymbolBase<MAX_INPUT, 1>
		{
		public:
			SumSymbol(const Array<Symbol*>& inputs)
			{
				mNumInputs = inputs.Size();

				for (int i = 0; i < mNumInputs; i++)
				{
					mInputs[i] = inputs[i];
				}
			}

			void Evaluate() override
			{
				Tensorf& ret = GetOutput();
				ret.Resize(mInputs[0]->GetOutput().Shape());
				ret.Clear();

				for (int i = 0; i < mNumInputs; i++)
				{
					ret += mInputs[i]->GetOutput();
				}
			}

			void Init()
			{
				Tensorf& output = GetOutput();
				output.Resize(mInputs[0]->GetOutput().Shape());
			}

			virtual int NumInputs() const override
			{
				return mNumInputs;
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			int mNumInputs;
		};

		class SumGradient : public SymbolBase<MAX_INPUT + 2, MAX_INPUT>
		{
		public:
			SumGradient(const Array<Symbol*>& inputs, Symbol* pSum, Symbol* pUpperGrads)
			{
				mNumInputs = inputs.Size();

				for (int i = 0; i < mNumInputs; i++)
				{
					mInputs[i] = inputs[i];
				}

				mInputs[mNumInputs] = pSum;
				mInputs[mNumInputs + 1] = pUpperGrads;
			}

			void Evaluate() override
			{
				Symbol* pSum = mInputs[mNumInputs];
				const Tensorf& upperGrads = mInputs[mNumInputs + 1]->GetOutput(pSum->GetGradientIndex());

				for (int i = 0; i < mNumInputs; i++)
				{
					GetOutput(i) = upperGrads;
					mInputs[i]->SetGradientIndex(i);
				}
			}

			virtual int NumInputs() const override
			{
				return mNumInputs + 2;
			}

		private:
			int mNumInputs;
		};
	}
}