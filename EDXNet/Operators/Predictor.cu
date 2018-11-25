#include "Predictor.h"

namespace EDX
{
	namespace DeepLearning
	{
		__global__ void PredictKernel(Tensorf ret, const Tensorf inputValue)
		{
			const int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i >= ret.LinearSize())
				return;

			float max = -1e32f;
			int index = -1;
			for (int j = 0; j < inputValue.Shape(1); j++)
			{
				float currentVal = inputValue(i, j);
				if (currentVal > max)
				{
					max = currentVal;
					index = j;
				}
			}

			ret[i] = index;
		}

		void InvokePredictKernel(Tensorf& ret, const Tensorf& inputValue)
		{
			const int linearSize = ret.LinearSize();
			const int blockDim = 256;
			const int gridDim = (linearSize + blockDim - 1) / blockDim;

			PredictKernel<<<gridDim, blockDim>>>(ret.Self(), inputValue.Self());
		}

		void Predictor::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();

			// Predicted result
			const int N = inputValue.Shape(0);
			Tensorf& output = GetOutput(0);
			output.Resize(N);

			if (inputValue.GetDeviceType() == CPU)
			{
				for (int i = 0; i < N; i++)
				{
					float max = Math::EDX_NEG_INFINITY;
					int index = -1;
					for (int j = 0; j < inputValue.Shape(1); j++)
					{
						float currentVal = inputValue(i, j);
						if (currentVal > max)
						{
							max = currentVal;
							index = j;
						}
					}

					output[i] = index;
				}
			}
			else
			{
				InvokePredictKernel(output, inputValue);
			}
		}
	}
}