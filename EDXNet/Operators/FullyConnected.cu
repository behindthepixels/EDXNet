#include "FullyConnected.h"


namespace EDX
{
	namespace DeepLearning
	{
		void FullyConnected::Evaluate()
		{
			Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& weights = mInputs[1]->GetOutput();
			const Tensorf& bias = mInputs[2]->GetOutput();

			const int N = inputValue.Shape(0);
			const int dim = inputValue.LinearSize() / N;
			Tensorf X = inputValue.GetWithShape(N, dim);

			Assertf(!X.Empty(), "Input data cannot be empty in fully connected layer.");

			GetOutput() = Tensorf::Dot(X, weights) + bias;
		}


		void FullyConnectedGradient::Evaluate()
		{
			Tensorf& inputValue = mInputs[0]->GetOutput();
			Tensorf& weights = mInputs[1]->GetOutput();
			const Tensorf& bias = mInputs[2]->GetOutput();
			const Tensorf& upperGrads = mInputs[3]->GetOutput();

			Tensorf& inputGrads = GetOutput(0);
			Tensorf& weightsGrads = GetOutput(1);
			Tensorf& biasesGrads = GetOutput(2);

			const int N = inputValue.Shape(0);
			const int dim = inputValue.LinearSize() / inputValue.Shape(0);
			Tensorf X = inputValue.GetWithShape(N, dim);

			inputGrads = Tensorf::Dot(upperGrads, weights.GetTransposed());
			weightsGrads = Tensorf::Dot(X.GetTransposed(), upperGrads);
			biasesGrads = Tensorf::Sum(upperGrads, { 0 });

			mInputs[0]->SetGradientIndex(0);
			mInputs[1]->SetGradientIndex(1);
			mInputs[2]->SetGradientIndex(2);
		}
	}
}