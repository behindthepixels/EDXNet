
template <typename Op>
__global__ void PoolingKernel(const int numThreads, const Tensorf inTensor,
	const int channels, const int height, const int width,
	const int pooledHeight, const int pooledWidth,
	const int kernelHeight, const int kernelWidth,
	const int padHeight, const int padWidth,
	const int strideHeight, const int strideWidth,
	Tensorf outTensor)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int pw = index % pooledWidth;
	const int ph = (index / pooledWidth) % pooledHeight;
	const int c = (index / pooledWidth / pooledHeight) % channels;
	const int n = index / pooledWidth / pooledHeight / channels;

	int hstart = ph * strideHeight - padHeight;
	int wstart = pw * strideWidth - padWidth;
	int hend = Math::Min(hstart + kernelHeight, height);
	int wend = Math::Min(wstart + kernelWidth, width);
	hstart = Math::Max(hstart, 0);
	wstart = Math::Max(wstart, 0);
	float pooledVal = Op::InitValue();

	for (int h = hstart; h < hend; ++h)
	{
		for (int w = wstart; w < wend; ++w)
		{
			const float val = inTensor(n, c, h, w);

			Op::Process(val, pooledVal);
		}
	}

	outTensor(n, c, ph, pw) = Op::Finalize(pooledVal, (hend - hstart) * (wend - wstart));
}

template <typename Op>
void InvokePoolingKernel(const Tensorf& inTensor,
	const int channels, const int height, const int width,
	const int pooledHeight, const int pooledWidth,
	const int kernelHeight, const int kernelWidth,
	const int padHeight, const int padWidth,
	const int strideHeight, const int strideWidth,
	Tensorf& outTensor)
{
	int numThreads = outTensor.LinearSize();

	const int blockDim = 256;
	const int gridDim = (numThreads + blockDim - 1) / blockDim;

	PoolingKernel<Op><<<gridDim, blockDim>>>(
		numThreads, inTensor.GetWithShape(inTensor.Shape()),
		channels, height, width,
		pooledHeight, pooledWidth,
		kernelHeight, kernelWidth,
		padHeight, padWidth,
		strideHeight, strideWidth,
		outTensor.GetWithShape(outTensor.Shape()));
}

template <typename Op>
__global__ void PoolingGradientKernel(const int numThreads, const Tensorf inputTensor, const Tensorf pooledTensor, const Tensorf upperGradsTensor,
	const int channels, const int height, const int width,
	const int pooledHeight, const int pooledWidth,
	const int kernelHeight, const int kernelWidth,
	const int padHeight, const int padWidth,
	const int strideHeight, const int strideWidth,
	Tensorf outTensor)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int pw = index % pooledWidth;
	const int ph = (index / pooledWidth) % pooledHeight;
	const int c = (index / pooledWidth / pooledHeight) % channels;
	const int n = index / pooledWidth / pooledHeight / channels;

	int hstart = ph * strideHeight - padHeight;
	int wstart = pw * strideWidth - padWidth;
	int hend = Math::Min(hstart + kernelHeight, height);
	int wend = Math::Min(wstart + kernelWidth, width);
	hstart = Math::Max(hstart, 0);
	wstart = Math::Max(wstart, 0);
	bool found = false;
	for (int h = hstart; h < hend; ++h)
	{
		for (int w = wstart; w < wend; ++w)
		{
			if (Op::Process(inputTensor(n, c, h, w),
				pooledTensor(n, c, ph, pw),
				upperGradsTensor(n, c, ph, pw),
				(hend - hstart) * (wend - wstart),
				outTensor(n, c, h, w)))
			{
				found = true;
				break;
			}
		}
		if (found)
			break;
	}
}

template <typename Op>
void InvokePoolingGradientKernel(const Tensorf& inputTensor, const Tensorf& pooledTensor, const Tensorf& upperGradsTensor,
	const int channels, const int height, const int width,
	const int pooledHeight, const int pooledWidth,
	const int kernelHeight, const int kernelWidth,
	const int padHeight, const int padWidth,
	const int strideHeight, const int strideWidth,
	Tensorf& outTensor)
{
	int numThreads = outTensor.LinearSize();

	const int blockDim = 256;
	const int gridDim = (numThreads + blockDim - 1) / blockDim;

	PoolingGradientKernel<Op><<<gridDim, blockDim>>>(
		numThreads,
		inputTensor.GetWithShape(inputTensor.Shape()),
		pooledTensor.GetWithShape(pooledTensor.Shape()),
		upperGradsTensor.GetWithShape(upperGradsTensor.Shape()),
		channels, height, width,
		pooledHeight, pooledWidth,
		kernelHeight, kernelWidth,
		padHeight, padWidth,
		strideHeight, strideWidth,
		outTensor.GetWithShape(outTensor.Shape()));
}