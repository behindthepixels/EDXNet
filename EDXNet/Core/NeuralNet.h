#pragma once

#include "Tensor.h"
#include "Math/SparseMatrix.h"

#include "Containers/Array.h"
#include "Containers/DimensionalArray.h"
#include "Containers/String.h"

#include "Core/SmartPointer.h"
#include "Core/Random.h"

#include "Math/EDXMath.h"

namespace EDX
{
	namespace DeepLearning
	{
		class Symbol
		{
		protected:
			bool mVisited = false;
			bool mUpdated = true;
			int mGradientIndex;

		public:
			virtual ~Symbol()
			{

			}

		public:
			template<typename Op>
			void Traverse(Op op)
			{
				if (Visited())
					return;

				SetVisited(true);

				for (int i = 0; i < NumInputs(); i++)
				{
					GetInput(i)->Traverse(op);
				}

				op(this);
			}

			virtual void Evaluate() = 0;
			virtual Symbol* Gradient(Symbol* pUpperGrads) const
			{
				return pUpperGrads;
			}

			virtual Tensorf& GetOutput(const int idx = 0) = 0;
			virtual const Tensorf& GetOutput(const int idx = 0) const = 0;

			virtual Symbol* GetInput(const int idx = 0) = 0;
			virtual const Symbol* GetInput(const int idx = 0) const = 0;

			virtual int NumInputs() const = 0;

			virtual void Init()
			{
				// No op
			}

			void SetGradientIndex(const int idx)
			{
				mGradientIndex = idx;
			}

			int GetGradientIndex() const
			{
				return mGradientIndex;
			}

			bool Visited() const
			{
				return mVisited;
			}

			void SetVisited(const bool val)
			{
				mVisited = val;
			}

			virtual void SetData(const Tensorf& inData)
			{
			}

			virtual void SetData(Tensorf&& inData)
			{
			}

			virtual bool IsOperator() const
			{
				return true;
			}

			virtual bool IsVariable() const
			{
				return false;
			}

			virtual bool Updated() const
			{
				return mUpdated;
			}

			void SetUpdated(const bool updated)
			{
				mUpdated = updated;
			}
		};

		template<int InputCount, int OutputCount>
		class SymbolBase : public Symbol
		{
		protected:
			Symbol* mInputs[InputCount == 0 ? 1 : InputCount];
			Tensorf mOutputs[OutputCount];

		public:
			SymbolBase()
			{
			}

			virtual Tensorf& GetOutput(const int idx = 0) override
			{
				return mOutputs[idx];
			}

			virtual const Tensorf& GetOutput(const int idx = 0) const override
			{
				return mOutputs[idx];
			}

			virtual Symbol* GetInput(const int idx = 0)
			{
				return mInputs[idx];
			}

			virtual const Symbol* GetInput(const int idx = 0) const
			{
				return mInputs[idx];
			}

			virtual int NumInputs() const override
			{
				return InputCount;
			}
		};

		class NeuralNet
		{
		private:
			Array<Symbol*> mSymbolArray;
			Symbol* mpRoot;
			Map<Symbol*, Symbol*> mSymbolToGradientMap;

		public:
			NeuralNet(Symbol* pRoot, const bool addGradient = false, const bool defaultInit = false)
				: mpRoot(pRoot)
			{
				// Flatten the symbolic graph in topological order
				mSymbolArray.Clear();
				NeuralNet::ResetVisited();

				TopologicalFlatten(mpRoot);

				if (addGradient)
					AddGradientSymbols();

				if (defaultInit)
				{
					for (auto it : mSymbolArray)
						it->Init();
				}
			}

			Symbol* GetRoot()
			{
				return mpRoot;
			}

			void Execute(const Array<Symbol*>& inSymbols)
			{
				mSymbolArray.Clear();
				NeuralNet::ResetVisited();

				for (auto it : inSymbols)
				{
					TopologicalFlatten(it);
				}

				NeuralNet::ResetVisited();
				for (auto it : inSymbols)
				{
					MarkUpdatedSymbols(it);
				}

				for (auto it : mSymbolArray)
				{
					if (it->Updated())
					{
						it->Evaluate();
					}
				}
			}

			Array<Symbol*> GetGradientSymbols(const Array<Symbol*>& inSymbols)
			{
				Array<Symbol*> ret;
				for (auto it : inSymbols)
				{
					ret.Add(mSymbolToGradientMap[it]);
				}

				return ret;
			}

			Array<Symbol*> GetVariableSymbols()
			{
				Array<Symbol*> ret;
				for (auto it : mSymbolArray)
				{
					if (it->IsVariable())
						ret.Add(it);
				}

				return ret;
			}

		private:
			void AddGradientSymbols();

			void TopologicalFlatten(Symbol* pRoot)
			{
				auto AppendSymbol = [this](Symbol* pSymbol)
				{
					mSymbolArray.Add(pSymbol);
				};

				pRoot->Traverse(AppendSymbol);
			}


			void MarkUpdatedSymbols(Symbol* pRoot)
			{
				auto MarkUpdated = [this](Symbol* pSymbol)
				{
					if (pSymbol->IsOperator())
					{
						for (int i = 0; i < pSymbol->NumInputs(); i++)
						{
							if (pSymbol->GetInput(i)->Updated())
							{
								pSymbol->SetUpdated(true);
								return;
							}
						}

						pSymbol->SetUpdated(false);
					}
				};

				pRoot->Traverse(MarkUpdated);
			}

		public:
			// Static methods for managing symbols
			template <typename T, typename... TArgs>
			static Symbol* Create(TArgs&&... args)
			{
				Symbol* ret = new T(Forward<TArgs>(args)...);
				mSymbols.Add(UniquePtr<Symbol>(ret));

				return ret;
			}

			static void ResetVisited()
			{
				for (auto& it : mSymbols)
					it->SetVisited(false);
			}

			static void Release()
			{
				mSymbols.Clear();
			}

		private:
			// Unique pointers for all symbols
			static Array<UniquePtr<Symbol>> mSymbols;

		};

		class Optimizer
		{
		public:
			virtual void Step(Tensorf& x, const Tensorf& dx) = 0;
		};


		template<typename Func>
		Tensorf NumericalGradientEval(Func f, Symbol* pData, const Tensorf& upperGrads, const float step = 1e-3f)
		{
			Tensorf& x = pData->GetOutput();

			Tensorf gradient;
			gradient.Resize(x.Shape());

			for (int i = 0; i < x.LinearSize(); i++)
			{
				float originalVal = x.Get(i);

				x.Set(i, originalVal + step);
				Tensorf positive = f();

				x.Set(i, originalVal - step);
				Tensorf negative = f();

				x.Set(i, originalVal);

				gradient.Set(i, Tensorf::Sum((positive - negative) * upperGrads).Get(0) / (2.0f * step));
			}

			return gradient;
		}
	}
}