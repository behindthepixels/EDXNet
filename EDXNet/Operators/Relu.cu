#include "Relu.h"

namespace EDX
{
	namespace DeepLearning
	{
		void Relu::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();

			GetOutput() = TensorExpr::ReluActivate(inputValue);
		}

		void ReluGradient::Evaluate()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& upperGrads = mInputs[1]->GetOutput();

			Tensorf& output = GetOutput();
			output = TensorExpr::ReluGradient(upperGrads, inputValue);

			mInputs[0]->SetGradientIndex(0);
		}
	}
}