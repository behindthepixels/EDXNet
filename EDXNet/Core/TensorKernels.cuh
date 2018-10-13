
template<typename ExpType, typename T>
__global__ void ExecuteExpressionKernel(const ExpType rhs, T* pData, const TensorParams tensorIndex)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= tensorIndex.LinearSize())
		return;

	pData[i] = rhs.Eval(i, tensorIndex);
}

template<typename ExpType, typename T>
void InvokeExecuteExpression(const ExpType& rhs, T* pData, const TensorParams& tensorIndex)
{
	const int linearSize = tensorIndex.LinearSize();
	const int blockDim = 64;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	ExecuteExpressionKernel<<<gridDim, blockDim>>>(rhs, pData, tensorIndex);
}


template<typename Op, typename TensorT>
__global__ void ElementWiseBinaryOpInplaceKernel(TensorT lhs, const TensorT rhs, Op op)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= lhs.LinearSize())
		return;

	TensorShape leftIndex;
	leftIndex.Resize(lhs.Dim());

	TensorShape rightIndex;
	rightIndex.Resize(rhs.Dim());

	TensorShape index = lhs.Index(i);
	for (int j = 0; j < lhs.Dim(); j++)
	{
		leftIndex[j] = index[j + lhs.Dim() - lhs.Dim()];
		if (leftIndex[j] >= lhs.Shape(j))
			leftIndex[j] = 0;
	}

	for (int j = 0; j < rhs.Dim(); j++)
	{
		rightIndex[j] = index[j + lhs.Dim() - rhs.Dim()];
		if (rightIndex[j] >= rhs.Shape(j))
			rightIndex[j] = 0;
	}

	lhs[i] = op(lhs(leftIndex), rhs(rightIndex));
}

template<typename Op, typename TensorT>
void InvokeElementWiseBinaryOpInplace(TensorT& lhs, const TensorT& rhs, Op op)
{
	const int linearSize = lhs.LinearSize();
	const int blockDim = 64;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;
	
	ElementWiseBinaryOpInplaceKernel<<<gridDim, blockDim>>>(lhs.Self(), rhs.Self(), op);
}