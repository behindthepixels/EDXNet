#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		enum class PoolingType
		{
			Max, Average
		};

		struct MaxPoolOp
		{
			static float InitValue()
			{
				return Math::EDX_NEG_INFINITY;
			}

			static void Process(const float x, float& y)
			{
				y = Math::Max(x, y);
			}

			static float Finalize(const float value, const int pooledSize)
			{
				return value;
			}
		};

		struct AvgPoolOp
		{
			static float InitValue()
			{
				return 0.0f;
			}

			static void Process(const float x, float& y)
			{
				y += x;
			}

			static float Finalize(const float value, const int pooledSize)
			{
				return value / float(pooledSize);
			}
		};

		template<typename Op>
		class Pooling : public SymbolBase<1, 1>
		{
		public:
			Pooling(Symbol* pInput,
				const TensorArray& kernelSize,
				const TensorArray& stride,
				const TensorArray& padding)
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
								float pooledVal = Op::InitValue();

								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int in_index = h * width + w;
										const float val = inputValue(n, c, h, w);

										Op::Process(val, pooledVal);
									}
								}

								output(n, c, ph, pw) = Op::Finalize(pooledVal, (hend - hstart) * (wend - wstart));
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
			TensorArray InferShape() const
			{
				TensorArray ret;

				const auto& inputShape = mInputs[0]->GetOutput().Shape();
				ret = inputShape;

				ret[2] = 1 + (inputShape[2] + 2 * mPadding[0] - mKernelSize[0]) / mStride[0];
				ret[3] = 1 + (inputShape[3] + 2 * mPadding[1] - mKernelSize[1]) / mStride[1];

				return ret;
			}

		private:
			TensorArray mKernelSize;
			TensorArray mStride;
			TensorArray mPadding;
		};

		struct MaxPoolGradOp
		{
			static bool Process(const float x, const float y, const float dy, const int pooledSize, float& dx)
			{
				if (x == y)
				{
					dx += dy;
					return true;
				}
				else
					return false;
			}
		};

		struct AvgPoolGradOp
		{
			static bool Process(const float x, const float y, const float dy, const int pooledSize, float& dx)
			{
				dx += dy / float(pooledSize);

				return false;
			}
		};

		template<typename Op>
		class PoolingGradient : public SymbolBase<3, 1>
		{
		public:
			PoolingGradient(Symbol* pInput,
				Symbol* pPooledOutput,
				Symbol* pUpperGradients,
				const TensorArray& kernelSize,
				const TensorArray& stride,
				const TensorArray& padding)
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

										if (Op::Process(inputValue(n, c, h, w),
											pooledValue(n, c, ph, pw),
											upperGradsAlias(n, c, ph, pw),
											(hend - hstart) * (wend - wstart),
											output(n, c, h, w)))
										{
											found = true;
											break;
										}
									}
									if (found)
										break;
								}
							}
						}
					}
				}
			}

		private:
			TensorArray InferShape() const
			{
				TensorArray ret;

				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const auto inputShape = inputValue.Shape();
				ret = inputShape;

				ret[2] = 1 + (inputShape[2] + 2 * mPadding[0] - mKernelSize[0]) / mStride[0];
				ret[3] = 1 + (inputShape[3] + 2 * mPadding[1] - mKernelSize[1]) / mStride[1];

				return ret;
			}

		private:
			TensorArray mKernelSize;
			TensorArray mStride;
			TensorArray mPadding;
		};


		using MaxPooling = Pooling<MaxPoolOp>;
		using AvgPooling = Pooling<AvgPoolOp>;
	}
}