#include "FullyConnected.h"


namespace EDX
{
	namespace DeepLearning
	{
		void FullyConnected::Init()
		{
			const Tensorf& inputValue = mInputs[0]->GetOutput();

			const int N = inputValue.Shape(0);
			const int dim = inputValue.LinearSize() / inputValue.Shape(0);

			Array<int> weightsShape = { dim, mNumHidden };

			Tensorf& weights = mInputs[1]->GetOutput();
			Tensorf& biases = mInputs[2]->GetOutput();

			weights = Tensorf::RandomNormalDistribution(0.1f, dim, mNumHidden);
			biases = Zeroes(mNumHidden);

			Tensorf& output = GetOutput();
			output.Resize(N, mNumHidden);
		}

		void FullyConnected::Evaluate()
		{
			Tensorf& inputValue = mInputs[0]->GetOutput();
			const Tensorf& weights = mInputs[1]->GetOutput();
			const Tensorf& bias = mInputs[2]->GetOutput();

			const int N = inputValue.Shape(0);
			const int dim = inputValue.LinearSize() / N;
			Tensorf X = inputValue.GetWithShape(N, dim);

			Assertf(!X.Empty(), "Input data cannot be empty in fully connected layer.");

			Tensorf& output = GetOutput();
			Tensorf::DotInplace(X, weights, &output);
			output += bias;
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

			Tensorf::DotInplace(upperGrads, weights.GetTransposed(), &inputGrads);
			Tensorf::DotInplace(X.GetTransposed(), upperGrads, &weightsGrads);
			Tensorf::SumInplace(upperGrads, &biasesGrads, { 0 });

			mInputs[0]->SetGradientIndex(0);
			mInputs[1]->SetGradientIndex(1);
			mInputs[2]->SetGradientIndex(2);
		}
	}
}