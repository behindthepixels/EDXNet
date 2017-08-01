#pragma once

#include "../Core/NeuralNet.h"

namespace EDX
{
	namespace DeepLearning
	{
		// Constant node represents a variable tensor in the computational graph
		// Typically they are tensors that need to be optimized
		class Variable : public SymbolBase<0, 1>
		{
		public:
			Variable() = default;
			Variable(const Tensorf& inData)
			{
				mOutputs[0] = inData;
			}
			Variable(Tensorf&& inData)
			{
				mOutputs[0] = Move(inData);
			}

			void Evaluate() override
			{
				// No op
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

			virtual bool IsVariable() const
			{
				return true;
			}
		};
	}
}