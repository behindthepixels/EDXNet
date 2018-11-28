#include "Pooling.h"

namespace EDX
{
	namespace DeepLearning
	{
		template <typename Op>
		__global__ void PoolingKernel(const int numThreads, const Tensorf inTensor,
			const int channels, const int height, const int width,
			const int pooledHeight, const int pooledWidth,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			Tensorf outTensor)
		{
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= numThreads)
				return;

			const int pw = index % pooledWidth;
			const int ph = (index / pooledWidth) % pooledHeight;
			const int c = (index / pooledWidth / pooledHeight) % channels;
			const int n = index / pooledWidth / pooledHeight / channels;

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
					const float val = inTensor(n, c, h, w);

					Op::Process(val, pooledVal);
				}
			}

			outTensor(n, c, ph, pw) = Op::Finalize(pooledVal, (hend - hstart) * (wend - wstart));
		}

		template <typename Op>
		void InvokePoolingKernel(const Tensorf& inTensor,
			const int channels, const int height, const int width,
			const int pooledHeight, const int pooledWidth,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			Tensorf& outTensor)
		{
			int numThreads = outTensor.LinearSize();

			const int blockDim = 256;
			const int gridDim = (numThreads + blockDim - 1) / blockDim;

			PoolingKernel<Op> << <gridDim, blockDim >> > (
				numThreads, inTensor.Self(),
				channels, height, width,
				pooledHeight, pooledWidth,
				kernelHeight, kernelWidth,
				padHeight, padWidth,
				strideHeight, strideWidth,
				outTensor.Self());
		}

		template <typename Op>
		__global__ void PoolingGradientKernel(const int numThreads, const Tensorf inputTensor, const Tensorf pooledTensor, const Tensorf upperGradsTensor,
			const int channels, const int height, const int width,
			const int pooledHeight, const int pooledWidth,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			Tensorf outTensor)
		{
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= numThreads)
				return;

			const int pw = index % pooledWidth;
			const int ph = (index / pooledWidth) % pooledHeight;
			const int c = (index / pooledWidth / pooledHeight) % channels;
			const int n = index / pooledWidth / pooledHeight / channels;

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
					if (Op::Process(inputTensor(n, c, h, w),
						pooledTensor(n, c, ph, pw),
						upperGradsTensor(n, c, ph, pw),
						(hend - hstart) * (wend - wstart),
						outTensor(n, c, h, w)))
					{
						found = true;
						break;
					}
				}
				if (found)
					break;
			}
		}

		template <typename Op>
		void InvokePoolingGradientKernel(const Tensorf& inputTensor, const Tensorf& pooledTensor, const Tensorf& upperGradsTensor,
			const int channels, const int height, const int width,
			const int pooledHeight, const int pooledWidth,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			Tensorf& outTensor)
		{
			int numThreads = pooledTensor.LinearSize();

			const int blockDim = 256;
			const int gridDim = (numThreads + blockDim - 1) / blockDim;

			PoolingGradientKernel<Op> << <gridDim, blockDim >> > (
				numThreads,
				inputTensor.Self(),
				pooledTensor.Self(),
				upperGradsTensor.Self(),
				channels, height, width,
				pooledHeight, pooledWidth,
				kernelHeight, kernelWidth,
				padHeight, padWidth,
				strideHeight, strideWidth,
				outTensor.Self());
		}

		template<typename Op>
		void PoolingEvaluate(const Tensorf& inputValue,
			Tensorf& output,
			const TensorShape& outputShape,
			const TensorShape& kernelSize,
			const TensorShape& stride,
			const TensorShape& padding)
		{
			const auto inputShape = inputValue.Shape();
			output.Resize(outputShape);

			const int height = inputShape[2], width = inputShape[3];
			const int pooledHeight = outputShape[2], pooledWidth = outputShape[3];
			const int kernelHeight = kernelSize[0], kernelWidth = kernelSize[1];
			const int padHeight = padding[0], padWidth = padding[1];
			const int strideHeight = stride[0], strideWidth = stride[1];

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
				InvokePoolingKernel<Op>(inputValue, outputShape[1], height, width, pooledHeight, pooledWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, output);
			}
		}

		void Pooling<MaxPoolOp>::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const auto outputShape = InferShape();

			PoolingEvaluate<MaxPoolOp>(inputValue, GetOutput(0), outputShape, mKernelSize, mStride, mPadding);
		}

		void Pooling<AvgPoolOp>::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const auto outputShape = InferShape();

			PoolingEvaluate<AvgPoolOp>(inputValue, GetOutput(0), outputShape, mKernelSize, mStride, mPadding);
		}

		template<typename Op>
		void PoolingGradientEvaluate(const Tensorf& inputValue,
			const Tensorf& pooledValue,
			Tensorf& upperGrads,
			Tensorf& output,
			const TensorShape& outputShape,
			const TensorShape& kernelSize,
			const TensorShape& stride,
			const TensorShape& padding)
		{
			const auto inputShape = inputValue.Shape();

			Tensorf upperGradsAlias = upperGrads.GetWithShape(outputShape);

			output.Resize(inputShape);
			output.Clear();

			const int height = inputShape[2], width = inputShape[3];
			const int pooledHeight = outputShape[2], pooledWidth = outputShape[3];
			const int kernelHeight = kernelSize[0], kernelWidth = kernelSize[1];
			const int padHeight = padding[0], padWidth = padding[1];
			const int strideHeight = stride[0], strideWidth = stride[1];

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
				InvokePoolingGradientKernel<Op>(inputValue, pooledValue, upperGradsAlias, outputShape[1], height, width, pooledHeight, pooledWidth, kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, output);
			}
		}

		void PoolingGradient<MaxPoolGradOp>::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& pooledValue = mInputs[1]->GetOutput();
			Tensorf& upperGrads = mInputs[2]->GetOutput();
			const auto outputShape = InferShape();

			PoolingGradientEvaluate<MaxPoolGradOp>(inputValue, pooledValue, upperGrads, GetOutput(0), outputShape, mKernelSize, mStride, mPadding);
		}

		void PoolingGradient<AvgPoolGradOp>::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& pooledValue = mInputs[1]->GetOutput();
			Tensorf& upperGrads = mInputs[2]->GetOutput();
			const auto outputShape = InferShape();

			PoolingGradientEvaluate<AvgPoolGradOp>(inputValue, pooledValue, upperGrads, GetOutput(0), outputShape, mKernelSize, mStride, mPadding);
		}
	}
}