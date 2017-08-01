#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{

		class Adam : public Optimizer
		{
		public:
			float mLearningRate = 1e-3f;
			float mBeta1 = 0.9f;
			float mBeta2 = 0.999f;
			float mEpsilon = 1e-8f;
			Map<Tensorf*, Tensorf> mMapM;
			Map<Tensorf*, Tensorf> mMapV;
			int mSteps = 0;

		public:
			Adam(const float learningRate = 1e-3f,
				const float beta1 = 0.9f,
				const float beta2 = 0.999f,
				const float eps = 1e-8f,
				const int steps = 0)
				: mLearningRate(learningRate)
				, mBeta1(beta1)
				, mBeta2(beta2)
				, mEpsilon(eps)
				, mSteps(steps)
			{

			}

			void SetLearningRate(const float learningRate)
			{
				mLearningRate = learningRate;
			}

			virtual void Step(Tensorf& x, const Tensorf& dx) override
			{
				if (!mMapM.Contains(&x))
				{
					mMapM.Add(&x, Tensorf::Zeroes(x.Shape()));
				}
				if (!mMapV.Contains(&x))
				{
					mMapV.Add(&x, Tensorf::Zeroes(x.Shape()));
				}

				Tensorf& m = mMapM[&x];
				Tensorf& v = mMapV[&x];

				mSteps++;
				m = Scalar(mBeta1) * m + Scalar(1 - mBeta1) * dx;
				v = Scalar(mBeta2) * v + Scalar(1 - mBeta2) * (dx * dx);
				Tensorf mt_hat = m / Scalar(1 - Math::Pow(mBeta1, mSteps));
				Tensorf vt_hat = v / Scalar(1 - Math::Pow(mBeta2, mSteps));
				x -= Scalar(mLearningRate) * mt_hat / (Tensorf::Sqrt(vt_hat + Scalar(mEpsilon)));
			}
		};
	}
}