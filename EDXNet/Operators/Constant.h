#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		// Constant node represents a constant tensor in the computational graph
		class Constant : public SymbolBase<0, 1>
		{
		public:
			Constant() = default;
			Constant(const Tensorf& inData)
			{
				mOutputs[0] = inData;
			}
			Constant(Tensorf&& inData)
			{
				mOutputs[0] = Move(inData);
			}

			void Evaluate() override
			{
				// No op
				mUpdated = false;
			}

			virtual int NumInputs() const override
			{
				return 0;
			}

			virtual Symbol* GetInput(const int idx = 0)
			{
				return nullptr;
			}

			virtual const Symbol* GetInput(const int idx = 0) const
			{
				return nullptr;
			}

			void SetData(const Tensorf& inData)
			{
				mOutputs[0] = inData;
			}

			void SetData(Tensorf&& inData)
			{
				mOutputs[0] = Move(inData);
			}

			virtual bool IsOperator() const
			{
				return false;
			}
		};
	}
}