#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class BatchNormalization : public SymbolBase<3, 6>
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

					// Output
					GetOutput(0) = scale * correctedX + bias;

					// Running mean
					GetOutput(1) = Scalar(mMomentum) * runningMean + Scalar(1.0f - mMomentum) * mean;
					// Running variance
					GetOutput(2) = Scalar(mMomentum) * runningVar + Scalar(1.0f - mMomentum) * variance;

					GetOutput(3) = centeredX;
					GetOutput(4) = invStdDiv;
					GetOutput(5) = variance;
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

			Symbol* Gradient(Symbol* pUpperGrads) const override;

		public:
			bool mTraining;
			float mMomentum;
		};

		class BatchNormalizationGradient : public SymbolBase<5, 3>
		{
		public:
			BatchNormalizationGradient(Symbol* pInput,
				Symbol* pScale,
				Symbol* pBias,
				Symbol* pBatchNorm,
				Symbol* pUpperGrads)
			{
				mInputs[0] = pInput;
				mInputs[1] = pScale;
				mInputs[2] = pBias;
				mInputs[3] = pBatchNorm;
				mInputs[4] = pUpperGrads;
			}

			void Evaluate() override
			{
				Tensorf& upperGrads = mInputs[4]->GetOutput();

				Tensorf& dBias = GetOutput(2);
				dBias = Tensorf::Sum(upperGrads, { 0 });

				const int N = mInputs[0]->GetOutput().Shape(0);
				Tensorf& centeredX = mInputs[3]->GetOutput(3);
				Tensorf& invStdDiv = mInputs[3]->GetOutput(4);
				Tensorf& variance = mInputs[3]->GetOutput(5);

				Tensorf& dScale = GetOutput(1);
				dScale = Tensorf::Sum(centeredX * invStdDiv * upperGrads, { 0 });

				Tensorf& scale = mInputs[1]->GetOutput();

				Tensorf& dx = GetOutput(0);
				dx = Scalar(1.0f / float(N)) * scale * invStdDiv *
					(Scalar(N) * upperGrads - dBias - centeredX / (variance + Scalar(1e-5f)) * Tensorf::Sum(upperGrads * centeredX, { 0 }));

				mInputs[0]->SetGradientIndex(0);
				mInputs[1]->SetGradientIndex(1);
				mInputs[2]->SetGradientIndex(2);
			}
		};
	}
}