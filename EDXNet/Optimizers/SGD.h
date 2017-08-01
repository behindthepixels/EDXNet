#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		class StochasticGradientDescent : public Optimizer
		{
		private:
			float mLearningRate;

		public:
			StochasticGradientDescent(float learningRate)
				: mLearningRate(learningRate)
			{

			}

			virtual void Step(Tensorf& x, const Tensorf& dx) override
			{
				x -= Scalar(mLearningRate) * dx;
			}
		};
	}
}