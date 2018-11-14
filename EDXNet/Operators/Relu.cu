#include "Relu.h"

namespace EDX
{
	namespace DeepLearning
	{
		void ReluGradient::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& upperGrads = mInputs[1]->GetOutput();

			Tensorf& output = GetOutput();
			output = ReluGradExp(upperGrads, inputValue);

			mInputs[0]->SetGradientIndex(0);
		}
	}
}