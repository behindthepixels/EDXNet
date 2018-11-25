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

			void Evaluate() override;

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

			void Evaluate() override;

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