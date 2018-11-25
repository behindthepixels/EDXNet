
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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
	const int blockDim = 256;
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
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;
	
	ElementWiseBinaryOpInplaceKernel<<<gridDim, blockDim>>>(lhs.Self(), rhs.Self(), op);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

/*
	This version adds multiple elements per thread sequentially.  This reduces the overall
	cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
	(Brent's Theorem optimization)

	Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
	In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
	If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <typename T, unsigned int blockSize, typename Op>
__global__ void ReduceKernel(T *g_idata, T *g_odata, unsigned int n, T initVal, Op op)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sharedMem = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T reduced = initVal;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		reduced = op(reduced, g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (i + blockSize < n)
			reduced = op(reduced, g_idata[i + blockSize]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sharedMem[tid] = reduced;
	cg::sync(cta);


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sharedMem[tid] = reduced = op(reduced, sharedMem[tid + 256]);
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128))
	{
		sharedMem[tid] = reduced = op(reduced, sharedMem[tid + 128]);
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64))
	{
		sharedMem[tid] = reduced = op(reduced, sharedMem[tid + 64]);
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64)
			reduced = op(reduced, sharedMem[tid + 32]);

		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset >>= 1)
		{
			reduced = op(reduced, tile32.shfl_down(reduced, offset));
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0)
		g_odata[blockIdx.x] = reduced;
}

template <typename T, typename Op>
void ReduceKernelSelection(int size, int threads, int blocks, T *pDataIn, T *pDataOut, T initVal, Op op)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int shareMemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	switch (threads)
	{
	case 512:
		ReduceKernel<T, 512, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case 256:
		ReduceKernel<T, 256, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case 128:
		ReduceKernel<T, 128, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case 64:
		ReduceKernel<T, 64, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case 32:
		ReduceKernel<T, 32, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case 16:
		ReduceKernel<T, 16, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case  8:
		ReduceKernel<T, 8, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case  4:
		ReduceKernel<T, 4, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case  2:
		ReduceKernel<T, 2, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;

	case  1:
		ReduceKernel<T, 1, Op><<<dimGrid, dimBlock, shareMemSize>>>(pDataIn, pDataOut, size, initVal, op);
		break;
	}
}

template <typename T, typename Op>
T* InvokeReduce(const T *pDataIn, const int n, T initVal, Op op)
{
	T* pDataCopy;
	cudaMalloc<T>(&pDataCopy, n * sizeof(T));
	cudaMemcpy(pDataCopy, pDataIn, n * sizeof(T), cudaMemcpyDeviceToDevice);

	const int maxThreads = 256;
	const int maxBlocks = 64;

	auto getNumBlocksAndThreads = [](int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
	{
		threads = (n < maxThreads * 2) ? Math::RoundUpPowOfTwo((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);

		blocks = Math::Min(maxBlocks, blocks);
	};

	int threads = 0;
	int blocks = 0;
	getNumBlocksAndThreads(n, maxBlocks, maxThreads, blocks, threads);

	T *pDataOut = nullptr;
	cudaMalloc<T>(&pDataOut, blocks * sizeof(T));

	// execute the kernel
	ReduceKernelSelection<T, Op>(n, threads, blocks, pDataCopy, pDataOut, initVal, op);

	if (blocks > 1)
	{
		int nextN = blocks;
		threads = 0;
		blocks = 0;
		getNumBlocksAndThreads(nextN, maxBlocks, maxThreads, blocks, threads);

		cudaMemcpy(pDataCopy, pDataOut, nextN * sizeof(T), cudaMemcpyDeviceToDevice);
		ReduceKernelSelection<T, Op>(nextN, threads, blocks, pDataCopy, pDataOut, initVal, op);
	}

	cudaFree(pDataCopy);

	return pDataOut;
}

template<typename TensorT, typename Op, typename T>
__global__ void TensorProjectionOpKernel(TensorT ret, const TensorT lhs, const TensorParams Params, const TensorShape axises, Op op, T initVal)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= ret.LinearSize())
		return;

	T reduced = initVal;

	TensorShape projIndex = Params.Index(i);

	do
	{
		reduced = op(reduced, lhs(projIndex));
	} while (lhs.IterateIndex(projIndex, axises));

	ret[i] = reduced;
}

template<typename TensorT, typename Op, typename T>
void InvokeTensorProjectionOp(TensorT& ret, const TensorT& lhs, const TensorParams& params, const TensorShape& axises, Op op, T initVal)
{
	const int linearSize = ret.LinearSize();
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	TensorProjectionOpKernel<<<gridDim, blockDim>>>(ret.Self(), lhs.Self(), params, axises, op, initVal);
}