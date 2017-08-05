#include "StyleTransferVGG19.h"

#include "../Operators/Constant.h"
#include "../Operators/Variable.h"
#include "../Operators/FullyConnected.h"
#include "../Operators/Convolution.h"
#include "../Operators/Pooling.h"
#include "../Operators/Relu.h"
#include "../Operators/Sum.h"
#include "../Operators/StyleTransfer.h"

#include "Windows/FileStream.h"

namespace EDX
{
	namespace DeepLearning
	{
		StyleTransferVGG19 StyleTransferVGG19::Create(const TCHAR* weightsPath, const float contentWeight, const float styleWeight, const float totalVariationWeight)
		{
			StyleTransferVGG19 net;

			VGG19Weights weights(false);
			weights.CreateConstant();

			FileStream stream(weightsPath, FileMode::Open);
			stream >> weights;
			stream.Close();

			return Create(weights, contentWeight, styleWeight, totalVariationWeight);
		}

		StyleTransferVGG19 StyleTransferVGG19::Create(const VGG19Weights& weights, const float contentWeight, const float styleWeight, const float totalVariationWeight)
		{
			StyleTransferVGG19 net;

			net.contentNet = VGG19::CreateForInference(weights);
			net.styleNet = VGG19::CreateForInference(weights);
			net.transferNet = VGG19::CreateForInference(weights, true);

			net.contentLoss = NeuralNet::Create<ContentLoss>(net.transferNet.conv4_2, net.contentNet.conv4_2);

			net.styleLoss1 = NeuralNet::Create<StyleLoss>(net.transferNet.relu1_1, net.styleNet.relu1_1);
			net.styleLoss2 = NeuralNet::Create<StyleLoss>(net.transferNet.relu2_1, net.styleNet.relu2_1);
			net.styleLoss3 = NeuralNet::Create<StyleLoss>(net.transferNet.relu3_1, net.styleNet.relu3_1);
			net.styleLoss4 = NeuralNet::Create<StyleLoss>(net.transferNet.relu4_1, net.styleNet.relu4_1);
			net.styleLoss5 = NeuralNet::Create<StyleLoss>(net.transferNet.relu5_1, net.styleNet.relu5_1);

			net.totalVariationLoss = NeuralNet::Create<TotalVariationLoss>(net.transferNet.input);

			net.styleLossSum = NeuralNet::Create<Sum>(Array<Symbol*>({ net.styleLoss1, net.styleLoss2, net.styleLoss3, net.styleLoss4, net.styleLoss5 }));

			net.transferLoss = NeuralNet::Create<StyleTransferLoss>(net.contentLoss, net.styleLossSum, net.totalVariationLoss, contentWeight, styleWeight, totalVariationWeight);

			return net;
		}
	}
}