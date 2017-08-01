#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
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
			float* data_im)
		{
			const int outputHeight = (height + 2 * padHeight - ((kernelHeight - 1) + 1)) / strideHeight + 1;
			const int outputWidth = (width + 2 * padWidth - ((kernelWidth - 1) + 1)) / strideWidth + 1;
			const int channelSize = height * width;

			for (int channel = channels; channel--; data_im += channelSize)
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
										data_im[inputRow * width + inputCol] += *dataCol;
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


		inline void Im2Col(const float* data_im,
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
			for (int channel = channels; channel--; data_im += channelSize)
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
										*(dataCol++) = data_im[inputRow * width + inputCol];
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

		class Convolution : public SymbolBase<3, 1>
		{
		public:
			Convolution(Symbol* pInput,
				Symbol* pKernels,
				Symbol* pBias,
				const Array<int>& kernelSize,
				const int numFilter,
				const Array<int>& stride,
				const Array<int>& padding)
				: mKernelSize(kernelSize)
				, mNumFilter(numFilter)
				, mStride(stride)
				, mPadding(padding)
			{
				mInputs[0] = pInput;
				mInputs[1] = pKernels;
				mInputs[2] = pBias;
			}

			void Evaluate() override
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

				Tensorf colBuffer;
				colBuffer.Resize(channel * kernelHeight * kernelWidth, outputHeight * outputWidth);

				Tensorf kernelMatrix = kernels.GetWithShape(mNumFilter, channel * kernelHeight * kernelWidth);

				for (int i = 0; i < N; i++)
				{
					Im2Col(inputValue.Data() + i * channel * height * width, channel, height, width, kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, colBuffer.Data());

					Tensorf slice = output.GetSlice(i).GetWithShape(mNumFilter, outputHeight * outputWidth);
					Tensorf::DotInplace(kernelMatrix, colBuffer, &slice);

					for (int c = 0; c < mNumFilter; c++)
					{
						Tensorf filterSlice = slice.GetSlice(c);
						filterSlice += bias[c];
					}
				}
			}

			void Init()
			{
				const auto& inputShape = mInputs[0]->GetOutput().Shape();

				const int N = inputShape[0];
				const int channel = inputShape[1];
				const int height = inputShape[2], width = inputShape[3];
				const int kernelHeight = mKernelSize[0], kernelWidth = mKernelSize[1];

				Tensorf& weights = mInputs[1]->GetOutput();
				Tensorf& biases = mInputs[2]->GetOutput();

				weights = Tensorf::RandomNormalDistribution(0.1f, mNumFilter, channel, kernelHeight, kernelWidth);
				biases = Scalar(0.1f) * Tensorf::Ones(mNumFilter);

				Tensorf& output = GetOutput();
				output.Resize(InferShape());
			}

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		private:
			Array<int> InferShape() const
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
			Array<int> mKernelSize;
			int mNumFilter;
			Array<int> mStride;
			Array<int> mPadding;
		};

		class ConvolutionGradient : public SymbolBase<5, 3>
		{
		public:
			ConvolutionGradient(Symbol* pInput,
				Symbol* pKernels,
				Symbol* pBiases,
				Symbol* pConvOut,
				Symbol* pUpperGradients,
				const Array<int>& kernelSize,
				const int numFilter,
				const Array<int>& stride,
				const Array<int>& padding)
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

			void Evaluate() override
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

				Tensorf colBuffer;
				colBuffer.Resize(channel * kernelHeight * kernelWidth, convOutHeight * convOutWidth);

				Tensorf kernelMatrix = kernels.GetWithShape(mNumFilter, channel * kernelHeight * kernelWidth);

				for (int i = 0; i < N; i++)
				{
					colBuffer.Clear();
					Tensorf upperGradsSlice = upperGrads.GetSlice(i).GetWithShape(mNumFilter, convOutHeight * convOutWidth);

					Tensorf::DotInplace(kernelMatrix.GetTransposed(), upperGradsSlice, &colBuffer);

					Col2Im(colBuffer.Data(), channel, height, width,
						kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth,
						inputGrads.Data() + i * channel * height * width);

					Im2Col(inputValue.Data() + i * channel * height * width, channel, height, width, kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, colBuffer.Data());


					Tensorf weightSlice = weightsGrads.GetWithShape(mNumFilter, channel * kernelHeight * kernelWidth);
					weightSlice += Tensorf::Dot(upperGradsSlice, colBuffer.GetTransposed());
				}

				Tensorf& biasesGrads = GetOutput(2);
				biasesGrads = Tensorf::Sum(upperGrads, { 0, 2, 3 });

				mInputs[0]->SetGradientIndex(0);
				mInputs[1]->SetGradientIndex(1);
				mInputs[2]->SetGradientIndex(2);
			}

		private:
			Array<int> InferShape() const
			{
				const Tensorf& weights = mInputs[1]->GetOutput();
				return weights.Shape();
			}


		private:
			Array<int> mKernelSize;
			int mNumFilter;
			Array<int> mStride;
			Array<int> mPadding;
		};
	}
}