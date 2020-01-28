#include "Convolution.h"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace EDX
{
	namespace DeepLearning
	{
		__global__ void Im2ColKernel(const int n, const float* dataIm,
			const int height, const int width,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			const int heightCol, const int widthCol,
			float* dataCol) {
			CUDA_KERNEL_LOOP(index, n)
			{
				const int hIndex = index / widthCol;
				const int colH = hIndex % heightCol;
				const int colW = index % widthCol;
				const int imC = hIndex / heightCol;
				const int colC = imC * kernelHeight * kernelWidth;
				const int offsetH = colH * strideHeight - padHeight;
				const int offsetW = colW * strideWidth - padWidth;
				float* pDataCol = dataCol;
				pDataCol += (colC * heightCol + colH) * widthCol + colW;
				const float* pDataIm = dataIm;
				pDataIm += (imC * height + offsetH) * width + offsetW;
				for (int i = 0; i < kernelHeight; ++i)
				{
					for (int j = 0; j < kernelWidth; ++j)
					{
						int imH = offsetH + i;
						int imW = offsetW + j;
						*pDataCol =
							(imH >= 0 && imW >= 0 && imH < height && imW < width) ?
							pDataIm[i * width + j] : 0.0f;
						pDataCol += heightCol * widthCol;
					}
				}
			}
		}

		/*!
		 * \brief DO NOT call this directly. Use wrapper function im2col() instead;
		 */
		inline void InvokeIm2ColKernel(const float* dataIm, const int channels,
			const int height, const int width,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			float* dataCol) {
			// We are going to launch channels * heightCol * widthCol kernels, each
			// kernel responsible for copying a single-channel grid.
			int heightCol = (height + 2 * padHeight -
				((kernelHeight - 1) + 1)) / strideHeight + 1;
			int widthCol = (width + 2 * padWidth -
				((kernelWidth - 1) + 1)) / strideWidth + 1;
			int numKernels = channels * heightCol * widthCol;

			const int blockDim = 256;
			const int gridDim = (numKernels + blockDim - 1) / blockDim;

			// NOLINT_NEXT_LINE(whitespace/operators)
			Im2ColKernel<<<gridDim, blockDim>>>(
					numKernels, dataIm, height, width, kernelHeight, kernelWidth, padHeight,
					padWidth, strideHeight, strideWidth, heightCol,
					widthCol, dataCol);
		}

		__global__ void Col2ImKernel(const int n, const float* dataCol,
			const int channels, const int height, const int width,
			const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			const int heightCol, const int widthCol,
			float* dataIm) {
			CUDA_KERNEL_LOOP(index, n)
			{
				float val = 0;
				const int imW = index % width + padWidth;
				const int imH = (index / width) % height + padHeight;
				const int imC = index / (width * height);
				int kernelExtentWidth = (kernelWidth - 1)/* * dilationWidth*/ + 1;
				int kernelExtentHeight = (kernelHeight - 1)/* * dilationHeight*/ + 1;
				// compute the start and end of the output
				const int wColStart =
					(imW < kernelExtentWidth) ? 0 : (imW - kernelExtentWidth) / strideWidth + 1;
				const int wColEnd = min(imW / strideWidth + 1, widthCol);
				const int hColStart =
					(imH < kernelExtentHeight) ? 0 : (imH - kernelExtentHeight) / strideHeight + 1;
				const int hColEnd = min(imH / strideHeight + 1, heightCol);
				// TODO(caffe): use LCM of stride and dilation to avoid unnecessary loops
				for (int colH = hColStart; colH < hColEnd; colH += 1)
				{
					for (int colW = wColStart; colW < wColEnd; colW += 1)
					{
						int h_k = (imH - colH * strideHeight);
						int w_k = (imW - colW * strideWidth);
						//if (h_k % dilationHeight == 0 && w_k % dilationWidth == 0)
						{
							//h_k /= dilationHeight;
							//w_k /= dilationWidth;
							int dataColIndex = (((imC * kernelHeight + h_k) * kernelWidth + w_k) *
								heightCol + colH) * widthCol + colW;
							val += dataCol[dataColIndex];
						}
					}
				}
				dataIm[index] = val;
			}
		}

		inline void InvokeCol2ImKernel(const float* dataCol, const int channels,
			const int height, const int width, const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth, const int strideHeight,
			const int strideWidth, float* dataIm) {
			int heightCol = (height + 2 * padHeight - ((kernelHeight - 1) + 1)) /
				strideHeight + 1;
			int widthCol = (width + 2 * padWidth - ((kernelWidth - 1) + 1)) /
				strideWidth + 1;
			int numKernels = channels * height * width;

			const int blockDim = 256;
			const int gridDim = (numKernels + blockDim - 1) / blockDim;

			// To avoid involving atomic operations, we will launch one kernel per
			// bottom dimension, and then in the kernel add up the top dimensions.
			// NOLINT_NEXT_LINE(whitespace/operators)
			Col2ImKernel<<<gridDim, blockDim>>>(
					numKernels, dataCol, channels, height, width, kernelHeight, kernelWidth,
					padHeight, padWidth, strideHeight, strideWidth,
					heightCol, widthCol, dataIm);
		}

		// Function uses casting from int to unsigned to compare if value of
		// parameter a is greater or equal to zero and lower than value of
		// parameter b. The b parameter is of type signed and is always positive,
		// therefore its value is always lower than 0x800... where casting
		// negative value of a parameter converts it to value higher than 0x800...
		// The casting allows to use one condition instead of two.
		inline bool ALEZeroAndALTB(int a, int b)
		{
			return static_cast<unsigned>(a) < static_cast<unsigned>(b);
		}


		inline void Col2Im(const float* dataCol,
			const int channels,
			const int height, const int width, const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			float* dataIm)
		{
			const int outputHeight = (height + 2 * padHeight - ((kernelHeight - 1) + 1)) / strideHeight + 1;
			const int outputWidth = (width + 2 * padWidth - ((kernelWidth - 1) + 1)) / strideWidth + 1;
			const int channelSize = height * width;

			for (int channel = channels; channel--; dataIm += channelSize)
			{
				for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++)
				{
					for (int kernelCol = 0; kernelCol < kernelWidth; kernelCol++)
					{
						int inputRow = -padHeight + kernelRow;
						for (int outputRows = outputHeight; outputRows; outputRows--)
						{
							if (!ALEZeroAndALTB(inputRow, height))
							{
								dataCol += outputWidth;
							}
							else
							{
								int inputCol = -padWidth + kernelCol;
								for (int outputCol = outputWidth; outputCol; outputCol--)
								{
									if (ALEZeroAndALTB(inputCol, width))
									{
										dataIm[inputRow * width + inputCol] += *dataCol;
									}
									dataCol++;
									inputCol += strideWidth;
								}
							}
							inputRow += strideHeight;
						}
					}
				}
			}
		}


		inline void Im2Col(const float* dataIm,
			const int channels,
			const int height, const int width, const int kernelHeight, const int kernelWidth,
			const int padHeight, const int padWidth,
			const int strideHeight, const int strideWidth,
			float* dataCol)
		{
			const int outputHeight = (height + 2 * padHeight -
				((kernelHeight - 1) + 1)) / strideHeight + 1;
			const int outputWidth = (width + 2 * padWidth -
				((kernelWidth - 1) + 1)) / strideWidth + 1;
			const int channelSize = height * width;
			for (int channel = channels; channel--; dataIm += channelSize)
			{
				for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++)
				{
					for (int kernelCol = 0; kernelCol < kernelWidth; kernelCol++)
					{
						int inputRow = -padHeight + kernelRow;
						for (int outputRows = outputHeight; outputRows; outputRows--)
						{
							if (!ALEZeroAndALTB(inputRow, height))
							{
								for (int outputCols = outputWidth; outputCols; outputCols--)
								{
									*(dataCol++) = 0;
								}
							}
							else
							{
								int inputCol = -padWidth + kernelCol;
								for (int outputCol = outputWidth; outputCol; outputCol--)
								{
									if (ALEZeroAndALTB(inputCol, width))
									{
										*(dataCol++) = dataIm[inputRow * width + inputCol];
									}
									else
									{
										*(dataCol++) = 0;
									}
									inputCol += strideWidth;
								}
							}
							inputRow += strideHeight;
						}
					}
				}
			}
		}

		void Convolution::Init()
		{
			const auto& inputShape = mInputs[0]->GetOutput().Shape();

			const int N = inputShape[0];
			const int channel = inputShape[1];
			const int height = inputShape[2], width = inputShape[3];
			const int kernelHeight = mKernelSize[0], kernelWidth = mKernelSize[1];

			Tensorf& weights = mInputs[1]->GetOutput();
			Tensorf& biases = mInputs[2]->GetOutput();

			weights = Tensorf::RandomNormalDistribution(0.1f, mNumFilter, channel, kernelHeight, kernelWidth);
			biases = Scalar(0.1f) * TensorExpr::Ones(mNumFilter);

			Tensorf& output = GetOutput();
			output.Resize(InferShape());
		}

		void Convolution::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			Tensorf& kernels = mInputs[1]->GetOutput();
			const Tensorf& bias = mInputs[2]->GetOutput();

			const auto inputShape = inputValue.Shape();
			const auto outputShape = InferShape();

			Tensorf& output = GetOutput();
			output.Resize(outputShape);
			output.Clear();

			const int N = inputShape[0];
			const int channel = inputShape[1];
			const int height = inputShape[2], width = inputShape[3];
			const int outputHeight = outputShape[2], outputWidth = outputShape[3];
			const int kernelHeight = mKernelSize[0], kernelWidth = mKernelSize[1];
			const int padHeight = mPadding[0], padWidth = mPadding[1];
			const int strideHeight = mStride[0], strideWidth = mStride[1];

			mColBuffer.Resize(channel * kernelHeight * kernelWidth, outputHeight * outputWidth);

			Tensorf kernelMatrix = kernels.GetWithShape(mNumFilter, channel * kernelHeight * kernelWidth);

			for (int i = 0; i < N; i++)
			{
				InvokeIm2ColKernel(inputValue.Data() + i * channel * height * width,
					channel, height, width,
					kernelHeight, kernelWidth,
					padHeight, padWidth,
					strideHeight, strideWidth,
					mColBuffer.Data());

				Tensorf slice = output.GetSlice(i).GetWithShape(mNumFilter, outputHeight * outputWidth);
				Tensorf::DotInplace(kernelMatrix, mColBuffer, &slice);

				for (int c = 0; c < mNumFilter; c++)
				{
					Tensorf filterSlice = slice.GetSlice(c);
					const Tensorf biasSlice = bias.GetSlice(c);
					filterSlice += biasSlice;
				}
			}
		}


		void ConvolutionGradient::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			Tensorf& kernels = mInputs[1]->GetOutput();
			const Tensorf& biases = mInputs[2]->GetOutput();
			const Tensorf& convOut = mInputs[3]->GetOutput();
			Tensorf& upperGrads = mInputs[4]->GetOutput();

			const auto inputShape = inputValue.Shape();
			const auto convOutShape = convOut.Shape();
			const auto outputShape = InferShape();

			Tensorf& inputGrads = GetOutput(0);
			inputGrads.Resize(inputShape);
			inputGrads.Clear();

			Tensorf& weightsGrads = GetOutput(1);
			weightsGrads.Resize(outputShape);
			weightsGrads.Clear();

			const int N = inputShape[0];
			const int channel = inputShape[1];
			const int height = inputShape[2], width = inputShape[3];
			const int outputHeight = outputShape[2], outputWidth = outputShape[3];
			const int convOutHeight = convOutShape[2], convOutWidth = convOutShape[3];
			const int kernelHeight = mKernelSize[0], kernelWidth = mKernelSize[1];
			const int padHeight = mPadding[0], padWidth = mPadding[1];
			const int strideHeight = mStride[0], strideWidth = mStride[1];

			mColBuffer.Resize(channel * kernelHeight * kernelWidth, convOutHeight * convOutWidth);

			Tensorf kernelMatrix = kernels.GetWithShape(mNumFilter, channel * kernelHeight * kernelWidth);

			for (int i = 0; i < N; i++)
			{
				mColBuffer.Clear();
				Tensorf upperGradsSlice = upperGrads.GetSlice(i).GetWithShape(mNumFilter, convOutHeight * convOutWidth);

				Tensorf::DotInplace(kernelMatrix.GetTransposed(), upperGradsSlice, &mColBuffer);

				InvokeCol2ImKernel(mColBuffer.Data(), channel, height, width,
					kernelHeight, kernelWidth,
					padHeight, padWidth, 
					strideHeight, strideWidth,
					inputGrads.Data() + i * channel * height * width);

				InvokeIm2ColKernel(inputValue.Data() + i * channel * height * width,
					channel, height, width,
					kernelHeight, kernelWidth,
					padHeight, padWidth,
					strideHeight, strideWidth,
					mColBuffer.Data());

				Tensorf weightSlice = weightsGrads.GetWithShape(mNumFilter, channel * kernelHeight * kernelWidth);
				Tensorf::DotInplace(upperGradsSlice, mColBuffer.GetTransposed(), &weightSlice, 1.0f, 1.0f);
			}

			Tensorf& biasesGrads = GetOutput(2);
			Tensorf::SumInplace(upperGrads, &biasesGrads, { 0, 2, 3 });

			mInputs[0]->SetGradientIndex(0);
			mInputs[1]->SetGradientIndex(1);
			mInputs[2]->SetGradientIndex(2);
		}
	}
}