#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class BatchNormalization : public SymbolBase<3, 3>
		{
		public:
			BatchNormalization(Symbol* pInput,
				Symbol* pScale,
				Symbol* pBias,
				const bool training,
				const float momentum)
			{
				mInputs[0] = pInput;
				mInputs[1] = pScale;
				mInputs[2] = pBias;

				mTraining = training;
				mMomentum = momentum;
			}

			void Evaluate() override
			{
				Tensorf& input = mInputs[0]->GetOutput();
				Tensorf& scale = mInputs[1]->GetOutput();
				Tensorf& bias = mInputs[2]->GetOutput();

				Tensorf& runningMean = this->GetOutput(1);
				Tensorf& runningVar = this->GetOutput(2);

				const int N = input.Shape(0);
				const int D = input.LinearSize() / N;

				runningMean.Resize(D);
				runningVar.Resize(D);

				Tensorf X = input.GetWithShape(N, D);

				if (mTraining)
				{
					Tensorf mean = Tensorf::Sum(X, { 0 }) / Scalar(float(N));
					Tensorf centeredX = X - mean;
					Tensorf centeredX2 = centeredX * centeredX;
					Tensorf variance = Tensorf::Sum(centeredX2, { 0 }) / Scalar(float(N));
					Tensorf stdDiv = Tensorf::Sqrt(variance + Scalar(1e-5f));
					Tensorf invStdDiv = Tensorf(1.0f) / stdDiv;
					Tensorf correctedX = centeredX * invStdDiv;

					GetOutput(0) = scale * correctedX + bias;

					GetOutput(1) = Scalar(mMomentum) * runningMean + Scalar(1.0f - mMomentum) * mean;
					GetOutput(2) = Scalar(mMomentum) * runningVar + Scalar(1.0f - mMomentum) * variance;
				}
				else
				{
					Tensorf correctedX = (X - runningMean) / Tensorf::Sqrt(runningVar + Scalar(1e-4f));
					GetOutput(0) = scale * correctedX + bias;
				}
			}

			void Init()
			{
				const auto& inputShape = mInputs[0]->GetOutput().Shape();

				Tensorf& output = GetOutput();
				output.Resize(inputShape);
			}

		public:
			bool mTraining;
			float mMomentum;
		};
	}
}