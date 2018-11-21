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
			TENSOR_INLINE static float InitValue()
			{
				return -1e32f;
			}

			TENSOR_INLINE static void Process(const float x, float& y)
			{
				y = Math::Max(x, y);
			}

			TENSOR_INLINE static float Finalize(const float value, const int pooledSize)
			{
				return value;
			}
		};

		struct AvgPoolOp
		{
			TENSOR_INLINE static float InitValue()
			{
				return 0.0f;
			}

			TENSOR_INLINE static void Process(const float x, float& y)
			{
				y += x;
			}

			TENSOR_INLINE static float Finalize(const float value, const int pooledSize)
			{
				return value / float(pooledSize);
			}
		};


#ifdef __CUDACC__
		#include "Pooling.cuh"
#endif

		template<typename Op>
		class Pooling : public SymbolBase<1, 1>
		{
		public:
			Pooling(Symbol* pInput,
				const TensorShape& kernelSize,
				const TensorShape& stride,
				const TensorShape& padding)
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

				if (inputValue.GetDeviceType() == CPU)
				{
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
				else if (inputValue.GetDeviceType() == GPU)
				{
#ifdef __CUDACC__
					InvokePoolingKernel<Op>(inputValue, outputShape[1], height, width, pooledHeight, pooledWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, output);
#endif
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
			TensorShape InferShape() const
			{
				TensorShape ret;

				const auto& inputShape = mInputs[0]->GetOutput().Shape();
				ret = inputShape;

				ret[2] = 1 + (inputShape[2] + 2 * mPadding[0] - mKernelSize[0]) / mStride[0];
				ret[3] = 1 + (inputShape[3] + 2 * mPadding[1] - mKernelSize[1]) / mStride[1];

				return ret;
			}

		private:
			TensorShape mKernelSize;
			TensorShape mStride;
			TensorShape mPadding;
		};

		struct MaxPoolGradOp
		{
			TENSOR_INLINE static bool Process(const float x, const float y, const float dy, const int pooledSize, float& dx)
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
			TENSOR_INLINE static bool Process(const float x, const float y, const float dy, const int pooledSize, float& dx)
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
				const TensorShape& kernelSize,
				const TensorShape& stride,
				const TensorShape& padding)
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

				if (inputValue.GetDeviceType() == CPU)
				{
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
									bool found = false;
									for (int h = hstart; h < hend; ++h)
									{
										for (int w = wstart; w < wend; ++w)
										{
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
				else if (inputValue.GetDeviceType() == GPU)
				{
#ifdef __CUDACC__
					InvokePoolingGradientKernel<Op>(inputValue, pooledValue, upperGradsAlias, outputShape[1], height, width, pooledHeight, pooledWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, output);
#endif
				}

			}

		private:
			TensorShape InferShape() const
			{
				TensorShape ret;

				const Tensorf& inputValue = mInputs[0]->GetOutput();
				const auto inputShape = inputValue.Shape();
				ret = inputShape;

				ret[2] = 1 + (inputShape[2] + 2 * mPadding[0] - mKernelSize[0]) / mStride[0];
				ret[3] = 1 + (inputShape[3] + 2 * mPadding[1] - mKernelSize[1]) / mStride[1];

				return ret;
			}

		private:
			TensorShape mKernelSize;
			TensorShape mStride;
			TensorShape mPadding;
		};


		using MaxPooling = Pooling<MaxPoolOp>;
		using AvgPooling = Pooling<AvgPoolOp>;
	}
}