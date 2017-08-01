#include "NeuralNet.h"

#include "../Operators/Sum.h"
#include "../Operators/Ones.h"

namespace EDX
{
	namespace DeepLearning
	{
		Array<UniquePtr<Symbol>> NeuralNet::mSymbols;

		void NeuralNet::AddGradientSymbols()
		{
			mSymbolToGradientMap.Clear();

			MultiMap<Symbol*, Symbol*> upperGradientsMap;
			upperGradientsMap.Add(mpRoot, NeuralNet::Create<Ones>(mpRoot));
			mpRoot->SetGradientIndex(0);

			// Traverse the graph in reverse topological order
			for (int i = mSymbolArray.Size() - 1; i >= 0; i--)
			{
				Symbol* pCurrentSymbol = mSymbolArray[i];

				Array<Symbol*> upperGradients;
				upperGradientsMap.MultiFind(pCurrentSymbol, upperGradients);

				Symbol* summedGrad = upperGradients.Size() > 1 ?
					NeuralNet::Create<Sum>(upperGradients) :
					upperGradients[0];

				Symbol* pGradientSymbol = pCurrentSymbol->Gradient(summedGrad);

				for (int j = 0; j < pCurrentSymbol->NumInputs(); j++)
				{
					Symbol* pInput = pCurrentSymbol->GetInput(j);
					upperGradientsMap.Add(pInput, pGradientSymbol);
				}

				mSymbolToGradientMap.Add(pCurrentSymbol, pGradientSymbol);
			}
		}
	}
}