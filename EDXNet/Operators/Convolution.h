#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Convolution : public SymbolBase<3, 1>
		{
		public:
			Convolution(Symbol* pInput,
				Symbol* pKernels,
				Symbol* pBias,
				const TensorShape& kernelSize,
				const int numFilter,
				const TensorShape& stride,
				const TensorShape& padding)
				: mKernelSize(kernelSize)
				, mNumFilter(numFilter)
				, mStride(stride)
				, mPadding(padding)
			{
				mInputs[0] = pInput;
				mInputs[1] = pKernels;
				mInputs[2] = pBias;
			}

			void Evaluate() override;
			void Init() override;

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			TensorShape InferShape() const
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const auto inputShape = inputValue.Shape();

				return{
					inputShape[0],
					mNumFilter,
					(inputShape[2] + 2 * mPadding[0] - ((mKernelSize[0] - 1) + 1)) / mStride[0] + 1,
					(inputShape[3] + 2 * mPadding[1] - ((mKernelSize[1] - 1) + 1)) / mStride[1] + 1
				};
			}


		private:
			TensorShape mKernelSize;
			int mNumFilter;
			TensorShape mStride;
			TensorShape mPadding;
		};

		class ConvolutionGradient : public SymbolBase<5, 3>
		{
		public:
			ConvolutionGradient(Symbol* pInput,
				Symbol* pKernels,
				Symbol* pBiases,
				Symbol* pConvOut,
				Symbol* pUpperGradients,
				const TensorShape& kernelSize,
				const int numFilter,
				const TensorShape& stride,
				const TensorShape& padding)
				: mKernelSize(kernelSize)
				, mNumFilter(numFilter)
				, mStride(stride)
				, mPadding(padding)
			{
				mInputs[0] = pInput;
				mInputs[1] = pKernels;
				mInputs[2] = pBiases;
				mInputs[3] = pConvOut;
				mInputs[4] = pUpperGradients;
			}

			void Evaluate() override;

		private:
			TensorShape InferShape() const
			{
				const Tensorf& weights = mInputs[1]->GetOutput();
				return weights.Shape();
			}


		private:
			TensorShape mKernelSize;
			int mNumFilter;
			TensorShape mStride;
			TensorShape mPadding;
		};
	}
}