#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Pooling : public SymbolBase<1, 1>
		{
		public:
			Pooling(Symbol* pInput,
				const Array<int>& kernelSize,
				const Array<int>& stride,
				const Array<int>& padding)
				: mKernelSize(kernelSize)
				, mStride(stride)
				, mPadding(padding)
			{
				mInputs[0] = pInput;
			}

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const auto inputShape = inputValue.Shape();
				const auto outputShape = InferShape();

				Tensorf& output = GetOutput(0);
				output.Resize(outputShape);

				const int height = inputShape[2], width = inputShape[3];
				const int pooledHeight = outputShape[2], pooledWidth = outputShape[3];
				const int kernelHeight = mKernelSize[0], kernelWidth = mKernelSize[1];
				const int padHeight = mPadding[0], padWidth = mPadding[1];
				const int strideHeight = mStride[0], strideWidth = mStride[1];

				for (int n = 0; n < outputShape[0]; ++n)
				{
					for (int c = 0; c < outputShape[1]; ++c)
					{
						for (int ph = 0; ph < pooledHeight; ++ph)
						{
							for (int pw = 0; pw < pooledWidth; ++pw)
							{
								int hstart = ph * strideHeight - padHeight;
								int wstart = pw * strideWidth - padWidth;
								int hend = Math::Min(hstart + kernelHeight, height);
								int wend = Math::Min(wstart + kernelWidth, width);
								hstart = Math::Max(hstart, 0);
								wstart = Math::Max(wstart, 0);
								float maxVal = Math::EDX_NEG_INFINITY;

								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int in_index = h * width + w;
										const float val = inputValue(n, c, h, w);
										if (val > maxVal)
										{
											maxVal = val;
										}
									}
								}

								output(n, c, ph, pw) = maxVal;
							}
						}
					}
				}
			}

			void Init()
			{
				const auto outputShape = InferShape();

				Tensorf& output = GetOutput(0);
				output.Resize(outputShape);
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			Array<int> InferShape() const
			{
				Array<int> ret;

				const auto& inputShape = mInputs[0]->GetOutput().Shape();
				ret = inputShape;

				ret[2] = 1 + (inputShape[2] + 2 * mPadding[0] - mKernelSize[0]) / mStride[0];
				ret[3] = 1 + (inputShape[3] + 2 * mPadding[1] - mKernelSize[1]) / mStride[1];

				return ret;
			}

		private:
			Array<int> mKernelSize;
			Array<int> mStride;
			Array<int> mPadding;
		};

		class PoolingGradient : public SymbolBase<3, 1>
		{
		public:
			PoolingGradient(Symbol* pInput,
				Symbol* pPooledOutput,
				Symbol* pUpperGradients,
				const Array<int>& kernelSize,
				const Array<int>& stride,
				const Array<int>& padding)
				: mKernelSize(kernelSize)
				, mStride(stride)
				, mPadding(padding)
			{
				mInputs[0] = pInput;
				mInputs[1] = pPooledOutput;
				mInputs[2] = pUpperGradients;
			}

			void Evaluate() override
			{
				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const Tensorf& pooledValue = mInputs[1]->GetOutput();
				Tensorf& upperGrads = mInputs[2]->GetOutput();
				const auto inputShape = inputValue.Shape();
				const auto outputShape = InferShape();

				Tensorf upperGradsAlias = upperGrads.GetWithShape(outputShape);

				Tensorf& output = GetOutput(0);
				output.Resize(inputShape);
				output.Clear();

				const int height = inputShape[2], width = inputShape[3];
				const int pooledHeight = outputShape[2], pooledWidth = outputShape[3];
				const int kernelHeight = mKernelSize[0], kernelWidth = mKernelSize[1];
				const int padHeight = mPadding[0], padWidth = mPadding[1];
				const int strideHeight = mStride[0], strideWidth = mStride[1];

				for (int n = 0; n < outputShape[0]; ++n)
				{
					for (int c = 0; c < outputShape[1]; ++c)
					{
						for (int ph = 0; ph < pooledHeight; ++ph)
						{
							for (int pw = 0; pw < pooledWidth; ++pw)
							{
								int hstart = ph * strideHeight - padHeight;
								int wstart = pw * strideWidth - padWidth;
								int hend = Math::Min(hstart + kernelHeight, height);
								int wend = Math::Min(wstart + kernelWidth, width);
								hstart = Math::Max(hstart, 0);
								wstart = Math::Max(wstart, 0);
								const int pool_index = ph * pooledWidth + pw;
								StaticArray<int, 4> max_idx;
								bool found = false;
								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int idx = h * width + w;
										if (inputValue(n, c, h, w) == pooledValue(n, c, ph, pw))
										{
											max_idx = { n,c,h,w };
											found = true;
											break;
										}
									}
									if (found)
										break;
								}

								// In the case where pad > 0 and kernel = 1, for example,
								// max_idx can be -1 reaching this step.
								if (!max_idx.Empty())
								{
									output(max_idx) += upperGradsAlias(n, c, ph, pw);
								}
							}
						}
					}
				}
			}

		private:
			Array<int> InferShape() const
			{
				Array<int> ret;

				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const auto inputShape = inputValue.Shape();
				ret = inputShape;

				ret[2] = 1 + (inputShape[2] + 2 * mPadding[0] - mKernelSize[0]) / mStride[0];
				ret[3] = 1 + (inputShape[3] + 2 * mPadding[1] - mKernelSize[1]) / mStride[1];

				return ret;
			}

		private:
			Array<int> mKernelSize;
			Array<int> mStride;
			Array<int> mPadding;
		};
	}
}