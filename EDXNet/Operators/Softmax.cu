#include "Softmax.h"

namespace EDX
{
	namespace DeepLearning
	{
		__global__ void SoftmaxKernel(const Tensorf probs, const Tensorf labels, Tensorf correctProbs)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index > correctProbs.LinearSize())
				return;

			correctProbs[index] = probs(index, int(labels[index]));
		}

		void Softmax::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& labels = mInputs[1]->GetOutput();

			mProbs = Tensorf::Exp(inputValue - Tensorf::Max(inputValue, { 1 }, true));
			mProbs /= Tensorf::Sum(mProbs, { 1 }, true);

			const int N = inputValue.Shape(0);
			mCorrectProbs.Resize(N);

			if (inputValue.GetDeviceType() == CPU)
			{
				for (int i = 0; i < N; i++)
				{
					mCorrectProbs[i] = mProbs(i, int(labels[i]));
				}
			}
			else if (inputValue.GetDeviceType() == GPU)
			{
				const int blockDim = 256;
				const int gridDim = (N + blockDim - 1) / blockDim;

				SoftmaxKernel<<<gridDim, blockDim>>>(mProbs.Self(), labels.Self(), correctProbs.Self());
			}

			// Loss
			Tensorf& output = GetOutput(0);
			output = Scalar(-1.0f) * Tensorf::Sum(Tensorf::Log(mCorrectProbs)) / Scalar(N);
		}

		__global__ void SoftmaxGradKernel(const Tensorf labels, Tensorf probs)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index > labels.LinearSize())
				return;

			probs(index, int(labels[index])) -= 1.0f;
		}

		void SoftmaxGradient::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& labels = mInputs[1]->GetOutput();

			mProbs = Tensorf::Exp(inputValue - Tensorf::Max(inputValue, { 1 }, true));
			mProbs /= Tensorf::Sum(mProbs, { 1 }, true);

			const int N = inputValue.Shape(0);
			if (inputValue.GetDeviceType() == CPU)
			{
				for (int i = 0; i < N; i++)
					mProbs(i, int(labels[i])) -= 1.0f;
			}
			else if (inputValue.GetDeviceType() == GPU)
			{
				const int blockDim = 256;
				const int gridDim = (N + blockDim - 1) / blockDim;

				SoftmaxGradKernel<<<gridDim, blockDim>>>(labels.Self(), mProbs.Self());
			}

			// dx
			Tensorf& output = GetOutput(0);
			output = mProbs / Scalar(N);
		}
	}
}