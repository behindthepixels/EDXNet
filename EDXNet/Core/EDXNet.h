#pragma once

#include "Tensor.h"
#include "NeuralNet.h"

#include "../Operators/Constant.h"
#include "../Operators/Variable.h"
#include "../Operators/BatchNormalization.h"
#include "../Operators/FullyConnected.h"
#include "../Operators/Convolution.h"
#include "../Operators/Dropout.h"
#include "../Operators/Pooling.h"
#include "../Operators/Relu.h"
#include "../Operators/Softmax.h"
#include "../Operators/Predictor.h"
#include "../Operators/Sum.h"
#include "../Operators/Ones.h"
#include "../Operators/StyleTransfer.h"

#include "../Optimizers/SGD.h"
#include "../Optimizers/Adam.h"
#include "../Optimizers/lbfgs.h"

#include "../Data/ImageNet.h"
#include "../Data/MINST.h"