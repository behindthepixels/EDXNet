#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "Containers/Array.h"
#include "Containers/Algorithm.h"
#include "Math/SparseMatrix.h"
#include "Core/Random.h"

#include "../OpenBLAS/include/cblas.h"
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "../OpenBLAS/include/lapacke.h"

#include <ppl.h>
using namespace concurrency;

#ifdef __CUDACC__
#define TENSOR_INLINE __forceinline __device__ __host__
#else
#define TENSOR_INLINE __forceinline
#endif

namespace EDX
{
	namespace DeepLearning
	{
		template<typename First, typename... Rest>
		struct AllIntegralType
		{
			enum { Value = And<IsIntegralType<First>, AllIntegralType<Rest...>>::Value };
		};

		template<typename Last>
		struct AllIntegralType<Last>
		{
			enum { Value = IsIntegralType<Last>::Value };
		};

		struct TensorShape
		{
			typedef int ElementType;

			static const int MaxArraySize = 5;
			int x[MaxArraySize];
			int mSize = 0;

			TENSOR_INLINE TensorShape()
			{
				mSize = x[0] = x[1] = x[2] = x[3] = x[4] = 0;
			}

			__forceinline TensorShape(std::initializer_list<int> InitList)
			{
				this->operator=(InitList);
			}

			__forceinline void operator = (std::initializer_list<int> InitList)
			{
				Assign(InitList.begin(), InitList.size());
			}

			TENSOR_INLINE bool operator == (const TensorShape& rhs) const
			{
				if (mSize != rhs.Size())
				{
					return false;
				}

				for (int i = 0; i < mSize; i++)
				{
					if (x[i] != rhs[i])
						return false;
				}

				return true;
			}

			TENSOR_INLINE bool operator != (const TensorShape& rhs) const
			{
				return !(*this == rhs);
			}


			TENSOR_INLINE int operator [] (const int idx) const
			{
				Assert(idx < MaxArraySize);
				return x[idx];
			}

			TENSOR_INLINE int& operator [] (const int idx)
			{
				Assert(idx < MaxArraySize);
				return x[idx];
			}


			TENSOR_INLINE int Size() const
			{
				return mSize;
			}

			TENSOR_INLINE bool Empty() const
			{
				return mSize == 0;
			}

			__forceinline void Clear(int32 Slack = 0)
			{
				ResizeZeroed(Slack);
			}

			TENSOR_INLINE void Resize(const int size)
			{
				Assert(size <= MaxArraySize);
				mSize = size;
			}

			__forceinline void ResizeZeroed(const int size)
			{
				Assert(size <= MaxArraySize);

				mSize = size;
				Memory::Memzero(x);
			}

			TENSOR_INLINE int* Data()
			{
				return x;
			}

			TENSOR_INLINE const int* Data() const
			{
				return x;
			}

			__forceinline void Assign(const int* Vals, const int32 Count)
			{
				mSize = x[0] = x[1] = x[2] = x[3] = x[4] = 0;

				Resize(Count);

				for (int i = 0; i < mSize; i++)
					x[i] = Vals[i];
			}

			TENSOR_INLINE bool Contains(const int val) const
			{
				for (int i = 0; i < mSize; i++)
				{
					if (x[i] == val)
						return true;
				}

				return false;
			}

			__forceinline bool Add(const int val)
			{
				Assert(mSize < 5);
				x[mSize++] = val;

				return false;
			}

			friend Stream& operator << (Stream& stream, TensorShape& A)
			{
				// Save array.
				stream << A.mSize;
				for (int32 i = 0; i < A.mSize; i++)
				{
					stream << A[i];
				}

				return stream;
			}

			friend Stream& operator >> (Stream& stream, TensorShape& A)
			{
				// Load array.
				int32 NewNum;
				stream >> NewNum;
				A.Clear(NewNum);
				for (int32 i = 0; i < NewNum; i++)
				{
					stream >> A[i];
				}

				return stream;
			}

		private:

			// --------------------------------------------------------------------------------------------
			// DO NOT USE DIRECTLY
			// STL-like iterators to enable range-based for loop support.
			// --------------------------------------------------------------------------------------------
			__forceinline friend int*			begin(TensorShape& shape) { return shape.Data(); }
			__forceinline friend const int*		begin(const TensorShape& shape) { return shape.Data(); }
			__forceinline friend int*			end(TensorShape& shape) { return shape.Data() + shape.Size(); }
			__forceinline friend const int*		end(const TensorShape& shape) { return shape.Data() + shape.Size(); }
		};

		//using TensorShape = StaticArray<int, 5>;

		class TensorParams
		{
		protected:
			TensorShape mShape;
			TensorShape mStrides;
			int mLinearSize;
			bool mTransposed = false;

		public:
			__forceinline TensorParams()
				: mLinearSize(0)
			{
			}

			template<typename... Shape>
			void Resize(Shape... shape)
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");

				mShape = { shape... };
				Resize_Common();
			}

			void Resize(const TensorShape& shape)
			{
				mShape = shape;
				Resize_Common();
			}

			template<typename... Shape>
			void Reshape(Shape... shape)
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");

#if defined(_DEBUG) && !defined(__CUDACC__) 
				auto oldSize = mLinearSize;
#endif

				mShape = { shape... };
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}

			void Reshape(const TensorShape& shape)
			{
#if defined(_DEBUG) && !defined(__CUDACC__) 
				auto oldSize = mLinearSize;
#endif

				mShape = shape;
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}
			
			void Transpose(const TensorShape& transposeDim = {})
			{
				if (transposeDim.Empty())
				{
					Algorithm::Reverse(mShape);
					Algorithm::Reverse(mStrides);
				}
				else
				{
					TensorShape shapeCopy = mShape;
					TensorShape strideCopy = mStrides;
					for (int i = 0; i < transposeDim.Size(); i++)
					{
						mShape[i] = shapeCopy[transposeDim[i]];
						mStrides[i] = strideCopy[transposeDim[i]];
					}
				}

				mTransposed = true;
			}

			template<typename... Index>
			TENSOR_INLINE int LinearIndex(Index... idx) const
			{
				constexpr int size = sizeof...(idx);
				Assertf(size == mShape.Size(), "Input index dimension does not match with array dimension.");

				int _idx[size] = { idx... };
				return LinearIndex<size>(_idx);
			}

			TENSOR_INLINE int LinearIndex(const TensorShape& idx) const
			{
				const int size = idx.Size();
				Assertf(size == mShape.Size(), "Input index dimension does not match with array dimension.");

				int ret = 0;
				for (auto i = 0; i < size; i++)
				{
					ret += idx[i] * mStrides[i];
				}

				Assert(ret < mLinearSize);
				return ret;
			}

			template<int N>
			TENSOR_INLINE int LinearIndex(const int idx[N]) const
			{
				Assertf(N == mShape.Size(), "Input index dimension does not match with array dimension.");

				int ret = 0;
				for (auto i = 0; i < N; i++)
				{
					ret += idx[i] * mStrides[i];
				}

				Assert(ret < mLinearSize);
				return ret;
			}

			TENSOR_INLINE TensorShape Index(int linearIdx) const
			{
				Assert(linearIdx < mLinearSize);

				TensorShape vRet;
				vRet.Resize(mShape.Size());
				for (int i = 0; i < mShape.Size(); i++)
				{
					vRet[i] = linearIdx / mStrides[i];
					linearIdx %= mStrides[i];
				}
				return vRet;
			}

			TENSOR_INLINE int LinearSize() const
			{
				return mLinearSize;
			}
			
			TENSOR_INLINE bool IndexRangeCheck(const TensorShape& index) const
			{
				for (int i = 0; i < mShape.Size(); i++)
				{
					if (index[i] >= mShape[i])
						return false;
				}
				return true;
			}

			TENSOR_INLINE void IterateIndex(TensorShape& index) const
			{
				for (int i = mShape.Size() - 1; i >= 0; i--)
				{
					index[i]++;

					if (index[i] < mShape[i])
						break;
					else
						index[i] = 0;
				}
			}

			TENSOR_INLINE int Shape(int iDim) const
			{
				Assert(iDim < mShape.Size());
				return mShape[iDim];
			}

			TENSOR_INLINE const TensorShape& Shape() const
			{
				return mShape;
			}

			TENSOR_INLINE TensorShape Shape()
			{
				return mShape;
			}

			TENSOR_INLINE int Stride(int iDim) const
			{
				Assert(iDim < mShape.Size());
				return mStrides[iDim];
			}

			__forceinline TensorParams GetSliceIndex(int subDim) const
			{
				Assert(subDim <= mShape.Size());
				TensorParams ret;

				if (subDim < mShape.Size())
				{
					ret.mShape.Assign(mShape.Data() + subDim, mShape.Size() - subDim);
				}
				else
				{
					ret.mShape = { 1 };
				}

				ret.Resize_Common();
				return ret;
			}

			__forceinline TensorParams GetSectionIndex(int num) const
			{
				Assert(num < mShape[0]);
				TensorParams ret;

				ret.mShape = mShape;
				ret.mShape[0] = num;

				ret.Resize_Common();
				return ret;
			}

			TENSOR_INLINE bool IsTransposed() const
			{
				return mTransposed;
			}

		private:
			void CalcStrides()
			{
				mStrides.Resize(mShape.Size());

				for (auto i = 0; i < mShape.Size(); i++)
				{
					mStrides[i] = 1;
					for (auto dim = mShape.Size() - 1; dim > i; dim--)
						mStrides[i] *= mShape[dim];
				}
			}

			__forceinline void Resize_Common()
			{
				CalcStrides();

				mLinearSize = mShape.Size() > 0 ? 1 : 0;
				for (auto i = 0; i < mShape.Size(); i++)
					mLinearSize *= mShape[i];
			}
		};

		static TensorShape BroadcastShape(const TensorShape& leftShape, const TensorShape& rightShape)
		{
			if (leftShape == rightShape) // Trivially broadcasted
			{
				return leftShape;
			}

			const int leftDim = leftShape.Size();
			const int rightDim = rightShape.Size();
			const auto& greaterShape = leftDim > rightDim ? leftShape : rightShape;
			const int retDim = greaterShape.Size();

			TensorShape ret;
			ret.Resize(retDim);

			int k = retDim - 1;
			for (int i = leftDim - 1, j = rightDim - 1;
				i >= 0 && j >= 0;
				i--, j--, k--)
			{
				Assertf(leftShape[i] == rightShape[j] ||
					leftShape[i] == 1 ||
					rightShape[j] == 1, "Tensor dimensions not aligned");

				ret[k] = Math::Max(leftShape[i], rightShape[j]);
			}

			while (k >= 0)
			{
				ret[k] = greaterShape[k];
				k--;
			}

			return ret;
		}

		#include "TemplateExpression.h"
		
#ifdef __CUDACC__
		#include "TensorKernels.cuh"
#endif

		template<class T>
		class Tensor : public TExp<Tensor<T>>
		{
		protected:
			T* mpData;
			TensorParams mParams;
			bool mReleaseData = true;

		public:
			static cublasHandle_t mCublasHandle;

		public:
			Tensor()
				: mpData(nullptr)
				, mReleaseData(true)
			{
			}

			virtual ~Tensor()
			{
				Free();
			}

			Tensor(const Tensor& rhs)
				: mpData(nullptr)
			{
				this->operator=(rhs);
			}

			Tensor(Tensor&& rhs)
				: mpData(nullptr)
			{
				this->operator=(Move(rhs));
			}

			Tensor(const T& val)
				: mpData(nullptr)
			{
				this->operator=(val);
			}

			Tensor& operator = (const Tensor& rhs)
			{
				Resize(rhs.Shape());
				//Memory::Memcpy(mpData, rhs.mpData, LinearSize() * sizeof(T));
				cudaMemcpy(mpData, rhs.mpData, LinearSize() * sizeof(T), cudaMemcpyDeviceToDevice);

				mReleaseData = rhs.mReleaseData;
				return *this;
			}

			Tensor& operator = (Tensor&& rhs)
			{
				Free();
				mParams = rhs.mParams;
				mpData = rhs.mpData;
				mReleaseData = rhs.mReleaseData;
				rhs.mpData = nullptr;
				return *this;
			}

			Tensor& operator = (const T val)
			{
				if (LinearSize() != 1)
				{
					Free();
					Resize(1);
				}
				*mpData = val;
				mReleaseData = true;
				return *this;
			}

			const Tensor<T> Self() const
			{
				return GetWithShape(Shape());
			}

			// ----------------------------------------------------------------
			// Constructor and assignment override for template expression
			// ----------------------------------------------------------------
			template<typename EType>
			Tensor(const TExp<EType>& rhs)
				: mpData(nullptr)
			{
				this->operator=(rhs);
			}

			template<typename EType>
			inline Tensor<T>& operator = (const TExp<EType>& rhs)
			{
				const EType& src = rhs.Self();
				Resize(src.Shape());

#ifdef __CUDACC__
				InvokeExecuteExpression(src, mpData, mParams);
#endif

				return *this;
			}

			TENSOR_INLINE T Eval(const int idx, const TensorParams& broadcastIndex) const
			{
				TensorShape selfIndex;
				selfIndex.Resize(Dim());

				TensorShape index = broadcastIndex.Index(idx);
				for (int j = 0; j < Dim(); j++)
				{
					selfIndex[j] = index[j + broadcastIndex.Shape().Size() - Dim()];
					if (selfIndex[j] >= Shape(j))
						selfIndex[j] = 0;
				}

				return this->operator()(selfIndex);
			}

			Tensor(NestedInitializerList<T, 1> initList)
				: mpData(nullptr)
			{
				this->operator=(initList);
			}

			Tensor(NestedInitializerList<T, 2> initList)
				: mpData(nullptr)
			{
				this->operator=(initList);
			}

			Tensor(NestedInitializerList<T, 3> initList)
				: mpData(nullptr)
			{
				this->operator=(initList);
			}

			Tensor(NestedInitializerList<T, 4> initList)
				: mpData(nullptr)
			{
				this->operator=(initList);
			}

			Tensor(NestedInitializerList<T, 5> initList)
				: mpData(nullptr)
			{
				this->operator=(initList);
			}

			Tensor& operator = (NestedInitializerList<T, 1> initList)
			{
				NestedInitializerListHelper<1>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 2> initList)
			{
				NestedInitializerListHelper<2>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 3> initList)
			{
				NestedInitializerListHelper<3>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 4> initList)
			{
				NestedInitializerListHelper<4>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 5> initList)
			{
				NestedInitializerListHelper<5>(initList);

				return *this;
			}

			template<int level>
			void NestedInitializerListHelper(NestedInitializerList<T, level> initList)
			{
				Resize(DeriveShapeFromNestedInitList<TensorShape>(initList));
				//InitListNestedCopy(Data(), initList);
				T* pTempData = new T[LinearSize()];
				T* pIter = pTempData;
				InitListNestedCopy(pIter, initList);

				cudaMemcpy(Data(), pTempData, LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
			}

			template<typename... TShape>
			void Resize(TShape... shape)
			{
				Resize({ shape... });
			}

			void Resize(const TensorShape& shape)
			{
				int newLinSize = Algorithm::Accumulate(shape, int(1), Algorithm::Multiply<>());
				if (LinearSize() != newLinSize)
				{
					Free();
					mParams.Resize(shape);

					//mpData = Memory::AlignedAlloc<T>(mParams.LinearSize());
					cudaMalloc<T>(&mpData, mParams.LinearSize() * sizeof(T));

					Assert(mpData);

					Clear();
				}
				else if (Shape() != shape)
				{
					Reshape(shape);
				}
			}

			void Assign(const T* pData, const TensorShape& shape)
			{
				Resize(shape);
				//Memory::Memcpy(mpData, pData, LinearSize() * sizeof(T));
				cudaMemcpy(mpData, pData, LinearSize() * sizeof(T));
			}

			void Assign(const T val, const TensorShape& shape)
			{
				Resize(shape);

				//// wrap raw pointer with a device_ptr 
				//thrust::device_ptr<T> dpData = thrust::device_pointer_cast(mpData);

				//// use device_ptr in Thrust algorithms
				//thrust::fill(dpData, dpData + LinearSize(), val);

				for (int i = 0; i < LinearSize(); i++)
					mpData[i] = val;
			}

			template<typename... Shape>
			Tensor Reshape(Shape... shape)
			{
				mParams.Reshape(shape...);
				return *this;
			}

			Tensor Reshape(const TensorShape& shape)
			{
				mParams.Reshape(shape);
				return *this;
			}

			Tensor GetTransposed(const TensorShape& transposeDim = {})
			{
				Tensor ret;
				ret.mParams = mParams;
				ret.mpData = mpData;
				ret.mReleaseData = false;

				ret.mParams.Transpose(transposeDim);

				return ret;
			}

			template<typename... Shape>
			Tensor GetWithShape(Shape... shape)
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");

				return GetWithShape({ shape... });
			}

			Tensor GetWithShape(const TensorShape& shape)
			{
				Tensor ret;
				ret.mParams = mParams;
				ret.mParams.Reshape(shape);
				ret.mpData = mpData;
				ret.mReleaseData = false;
				return ret;
			}

			template<typename... Shape>
			const Tensor GetWithShape(Shape... shape) const
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");

				return GetWithShape({ shape... });
			}

			const Tensor GetWithShape(const TensorShape& shape) const
			{
				Tensor ret;
				ret.mParams = mParams;
				ret.mParams.Reshape(shape);
				ret.mpData = mpData;
				ret.mReleaseData = false;
				return ret;
			}

			template<typename... Index>
			Tensor GetSlice(Index... index)
			{
				static_assert(AllIntegralType<Index...>::Value, "All parameters have to be integral type.");

				return GetSlice({ index... });
			}

			Tensor GetSlice(const TensorShape& index)
			{
				Tensor ret;

				TensorShape filledIndex = index;
				int numComponentToFill = Dim() - index.Size();
				while (numComponentToFill--)
				{
					filledIndex.Add(0);
				}

				ret.mpData = &this->operator()(filledIndex);
				ret.mParams = mParams.GetSliceIndex(index.Size());
				ret.mReleaseData = false;
				return ret;
			}


			template<typename... Index>
			const Tensor GetSlice(Index... index) const
			{
				static_assert(AllIntegralType<Index...>::Value, "All parameters have to be integral type.");

				return GetSlice({ index... });
			}

			const Tensor GetSlice(const TensorShape& index) const
			{
				Tensor ret;

				TensorShape filledIndex = index;
				int numComponentToFill = Dim() - index.Size();
				while (numComponentToFill--)
				{
					filledIndex.Add(0);
				}

				ret.mpData = &this->operator()(filledIndex);
				ret.mParams = mParams.GetSliceIndex(index.Size());
				ret.mReleaseData = false;
				return ret;
			}

			Tensor GetSection(const int from, const int to)
			{
				Assert(from < to);

				Tensor ret;

				ret.mpData = mpData + from * mParams.Stride(0);
				ret.mParams = mParams.GetSectionIndex(to - from);
				ret.mReleaseData = false;

				return ret;
			}

			const Tensor GetSection(const int from, const int to) const
			{
				Assert(from < to);

				Tensor ret;

				ret.mpData = mpData + from * mParams.Stride(0);
				ret.mParams = mParams.GetSectionIndex(to - from);
				ret.mReleaseData = false;

				return ret;
			}

			__forceinline bool IsTransposed() const
			{
				return mParams.IsTransposed();
			}

			__forceinline void Clear()
			{
				//Memory::SafeClear(mpData, mParams.LinearSize());

				if (mpData)
					cudaMemset(mpData, 0, mParams.LinearSize() * sizeof(T));
			}

			template<typename... Index>
			TENSOR_INLINE int LinearIndex(Index... idx) const
			{
				return mParams.LinearIndex(idx...);
			}
			TENSOR_INLINE int LinearIndex(const TensorShape& idx) const
			{
				return mParams.LinearIndex(idx);
			}
			TENSOR_INLINE virtual TensorShape Index(int linearIdx) const
			{
				return mParams.Index(linearIdx);
			}
			TENSOR_INLINE int LinearSize() const
			{
				return mParams.LinearSize();
			}
			TENSOR_INLINE bool IndexRangeCheck(const TensorShape& index) const
			{
				return mParams.IndexRangeCheck(index);
			}
			TENSOR_INLINE void IterateIndex(TensorShape& index) const
			{
				return mParams.IterateIndex(index);
			}
			TENSOR_INLINE int Shape(int iDim) const
			{
				return mParams.Shape(iDim);
			}
			TENSOR_INLINE const TensorShape& Shape() const
			{
				return mParams.Shape();
			}
			TENSOR_INLINE TensorShape Shape()
			{
				return mParams.Shape();
			}
			TENSOR_INLINE int Dim() const
			{
				return Shape().Size();
			}
			TENSOR_INLINE int Stride(int iDim) const
			{
				return mParams.Stride(iDim);
			}
			TENSOR_INLINE bool Empty() const
			{
				return LinearSize() == 0;
			}

			template<typename... Index>
			TENSOR_INLINE T& operator () (Index... idx)
			{
				return mpData[LinearIndex(idx...)];
			}

			template<typename... Index>
			TENSOR_INLINE const T& operator () (Index... idx) const
			{
				return mpData[LinearIndex(idx...)];
			}

			TENSOR_INLINE T& operator [] (const TensorShape& idx)
			{
				return mpData[LinearIndex(idx)];
			}
			TENSOR_INLINE const T& operator [] (const TensorShape& idx) const
			{
				return mpData[LinearIndex(idx)];
			}
			TENSOR_INLINE T& operator [] (const int idx)
			{
				Assert(idx < mParams.LinearSize());
				return mpData[idx];
			}
			TENSOR_INLINE const T& operator [] (const int idx) const
			{
				Assert(idx < mParams.LinearSize());
				return mpData[idx];
			}
			TENSOR_INLINE const T* Data() const
			{
				return mpData;
			}
			TENSOR_INLINE T* Data()
			{
				return mpData;
			}

			void Free()
			{
				if (mReleaseData)
				{
					//Memory::SafeFree(mpData);
					if (mpData)
						cudaFree(mpData);
					mpData = nullptr;
				}
			}

			// ----------------------------------------------------------------
			// Serialization operator
			// ----------------------------------------------------------------
			friend Stream& operator << (Stream& stream, Tensor& A)
			{
				// Save array.
				stream << A.Shape();
				stream.ByteOrderWrite(A.Data(), A.LinearSize() * sizeof(T));

				return stream;
			}

			friend Stream& operator >> (Stream& stream, Tensor& A)
			{
				// Load array.
				TensorShape newShape;
				stream >> newShape;
				A.Resize(newShape);
				stream.ByteOrderRead(A.Data(), A.LinearSize() * sizeof(T));

				return stream;
			}

		public:
			// ----------------------------------------------------------------
			// Common utilities for tensors
			// ----------------------------------------------------------------
			static Tensor<T> LinSpace(const T& start, const T& stop, const int& numSamples)
			{
				Tensor<T> ret;
				ret.Resize(numSamples);

				T step = (stop - start) / T(numSamples - 1);

				for (int i = 0; i < numSamples; i++)
					ret[i] = start + i * step;

				return ret;
			}

			static Tensor<T> ArrayRange(const T& start, const T& stop, const T& step = 1)
			{
				Tensor<T> ret;

				if (stop <= start)
					return ret;

				int numSamples = (stop - start) / step;
				ret.Resize(numSamples);

				for (int i = 0; i < numSamples; i++)
					ret[i] = start + i * step;

				return ret;
			}

			static Tensor<T> ArrayRange(const T& stop)
			{
				T start = 0;
				return ArrayRange(0, stop, 1);
			}

			template<typename... Shape>
			static TConstantExp Zeroes(Shape&&... shape)
			{
				return TConstantExp(0.0f, Forward<Shape>(shape)...);
			}

			template<typename... Shape>
			static TConstantExp Ones(Shape&&... shape)
			{

				return TConstantExp(1.0f, Forward<Shape>(shape)...);
			}

			static Tensor<T> Identity(const int N)
			{
				Tensor<T> ret;
				ret.Resize(N, N);

				for (int i = 0; i < N; i++)
				{
					ret(i, i) = T(1);
				}

				return ret;
			}

			static Tensor<T> Normalize(const Tensor<T>& inTensor)
			{
				const Tensor<T> tensorSqr = inTensor * inTensor;
				Tensor<T> denorm = Tensor<T>::Sqrt(Tensor<T>::Sum(tensorSqr));

				return inTensor / denorm;
			}

			static Tensor<T> Dot(const Tensor<T>& lhs, const Tensor<T>& rhs)
			{
				const TensorShape& leftShape = lhs.Shape();
				const TensorShape& rightShape = rhs.Shape();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

				Tensor<T> ret;
				ret.Resize(leftShape[0], rightShape[1]);

				DotInplace(lhs, rhs, &ret);

				return ret;
			}

			static void DotInplace(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>* pResult)
			{
				const TensorShape& leftShape = lhs.Shape();
				const TensorShape& rightShape = rhs.Shape();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

				//cblas_sgemm(CblasRowMajor,
				//	lhs.IsTransposed() ? CblasTrans : CblasNoTrans,
				//	rhs.IsTransposed() ? CblasTrans : CblasNoTrans,
				//	leftShape[0],
				//	rightShape[1],
				//	leftShape[1],
				//	1.0f,
				//	lhs.Data(),
				//	lhs.IsTransposed() ? lhs.Shape(0) : lhs.Shape(1),
				//	rhs.Data(),
				//	rhs.IsTransposed() ? rhs.Shape(0) : rhs.Shape(1),
				//	0.0f,
				//	pResult->Data(),
				//	pResult->Shape(1));

				const float alpha = 1.0f;
				const float beta = 0.0f;

				cublasSgemm(mCublasHandle,
					rhs.IsTransposed() ? CUBLAS_OP_T : CUBLAS_OP_N,
					lhs.IsTransposed() ? CUBLAS_OP_T : CUBLAS_OP_N,
					rightShape[1],
					leftShape[0],
					leftShape[1],
					&alpha,
					rhs.Data(),
					rhs.IsTransposed() ? rhs.Shape(0) : rhs.Shape(1),
					lhs.Data(),
					lhs.IsTransposed() ? lhs.Shape(0) : lhs.Shape(1),
					&beta,
					pResult->Data(),
					pResult->Shape(1));
			}
			
			static Tensor<T> Dot(const SparseMatrix<T>& lhs, const Tensor<T>& rhs)
			{
				const TensorShape& rightShape = rhs.Shape();

				Assertf(rightShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");
				Assertf(!(rightShape.Size() == 2 && lhs.n != rightShape[0]), "Dimension mismatch for tensor multiply.");

				Tensor<T> ret;
				ret.Resize(lhs.n, rightShape[1]);

				DotInplace(lhs, rhs, &ret);

				return ret;
			}

			static void DotInplace(const SparseMatrix<T>& lhs, const Tensor<T>& rhs, Tensor<T>* pResult)
			{
				const TensorShape& rightShape = rhs.Shape();

				Assertf(rightShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");
				Assertf(!(rightShape.Size() == 2 && lhs.n != rightShape[0]), "Dimension mismatch for tensor multiply.");

				parallel_for(0, (int)lhs.n, [&](int i)
				{
					pResult[i] = 0;
					for (int j = lhs.rowstart[i]; j < lhs.rowstart[i + 1]; ++j)
					{
						pResult[i] += lhs.value[j] * rhs[lhs.colindex[j]];
					}
				});
			}

			static Tensor<T> Transpose(const Tensor<T>& inTensor, const TensorShape& transposeDim = {})
			{
				const TensorShape& inShape = inTensor.Shape();

				TensorShape transposedShape = inShape;
				if (transposeDim.Empty())
				{
					Algorithm::Reverse(transposedShape);
				}
				else
				{
					for (int i = 0; i < transposeDim.Size(); i++)
					{
						transposedShape[i] = inShape[transposeDim[i]];
					}
				}

				Tensor<T> ret;
				ret.Resize(transposedShape);

				TensorShape index;
				index.ResizeZeroed(inTensor.Dim());
				for (int i = 0; i < inTensor.LinearSize(); i++, inTensor.IterateIndex(index))
				{
					TensorShape transposedIndex = index;

					if (transposeDim.Empty())
					{
						Algorithm::Reverse(transposedIndex);
					}
					else
					{
						for (int i = 0; i < transposeDim.Size(); i++)
						{
							transposedIndex[i] = index[transposeDim[i]];
						}
					}

					ret(transposedIndex) = inTensor(index);
				}

				return ret;
			}

			static Tensor<T> Inverse(const Tensor<T>& inTensor)
			{
				Assertf(inTensor.Dim() == 2 && inTensor.Shape(0) == inTensor.Shape(1),
					"Matrix inversion dimension mismatch.");

				const int N = inTensor.Shape(0);

				Tensor<T> ret = inTensor;
				float* A = ret.Data();

				int* ipiv = new int[N + 1];

				lapack_int code;
				code = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
					N,
					N,
					A,
					N,
					ipiv);

				if (code != 0)
					return ret;

				code = LAPACKE_sgetri(LAPACK_ROW_MAJOR,
					N,
					A,
					N,
					ipiv);

				delete[] ipiv;

				return ret;
			}


			template<typename TParam>
			static inline TUnaryExp<ExpOp, TParam> Exp(const TExp<TParam>& param)
			{
				return ExponentExp(param);
			}

			template<typename TParam>
			static inline TUnaryExp<SqrtOp, TParam> Sqrt(const TExp<TParam>& param)
			{
				return SqrtExp(param);
			}

			template<typename TParam>
			static inline TUnaryExp<SquareOp, TParam> Square(const TExp<TParam>& param)
			{
				return SquareExp(param);
			}

			template<typename TParam>
			static inline TUnaryExp<LogOp, TParam> Log(const TExp<TParam>& param)
			{
				return LogExp(param);
			}

			template<typename TParam>
			static inline TUnaryExp<AbsOp, TParam> Abs(const TExp<TParam>& param)
			{
				return AbsExp(param);
			}

			template<typename TParam>
			static inline TUnaryExp<ReluOp, TParam> ReluActivate(const TExp<TParam>& param)
			{
				return ReluActivateExp(param);
			}

			static Tensor<T> Sum(const Tensor<T>& inTensor, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Plus<T>(), T(0));
			}

			static Tensor<T> Product(const Tensor<T>& inTensor, const TensorShape& axises = { -1 })
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Multiply<>(), T(1));
			}

			static Tensor<T> Max(const Tensor<T>& inTensor, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				struct MaxOp
				{
					// Universal reference and late binding
					__forceinline T operator()(const T& lhs, const T& rhs)
					{
						return lhs > rhs ? lhs : rhs;
					}
				};

				return ProjectionOp(inTensor, axises, keepDim, MaxOp(), T(Math::EDX_NEG_INFINITY));
			}

			static Tensor<T> Mean(const Tensor<T>& X, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				Tensor<T> ret = Sum(X, axises);

				float invDivisor = ret.LinearSize() / float(X.LinearSize());
				ret *= invDivisor;

				return ret;
			}

			static Tensor<T> StandardDeviation(const Tensor<T>& X, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				Tensor<T> mean = Mean(X, axises, keepDim);
				Tensor<T> centeredX = X - mean;

				Tensor<T> variance = Tensorf::Mean(centeredX * centeredX, axises, keepDim);

				return Sqrt(variance + Scalar(1e-5f));
			}

			template<typename... Shape>
			static Tensor<T> RandomInt(const int high, Shape... shape)
			{
				Tensor<T> ret;
				ret.Resize(shape...);

				RandomGen random;
				for (auto& it : ret)
				{
					it = random.UnsignedInt() % high;
				}

				return ret;
			}

			template<typename... Shape>
			static Tensor<T> RandomFloat(Shape... shape)
			{
				Tensor<T> ret;
				ret.Resize(shape...);

				RandomGen random;
				for (auto& it : ret)
				{
					it = random.Float();
				}

				return ret;
			}

			template<typename... Shape>
			static Tensor<T> RandomNormalDistribution(const float std, Shape... shape)
			{
				Tensor<T> ret;
				ret.Resize(shape...);

				RandomGen random;
				for (auto& it : ret)
				{
					it = random.GaussFloat(0.0f, std);
				}

				return ret;
			}

			void operator += (const Tensor<T>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Plus<>());
			}
			void operator -= (const Tensor<T>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Substract<>());
			}

			void operator *= (const Tensor<T>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Multiply<>());
			}
			void operator /= (const Tensor<T>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Divide<>());
			}

		private:

			template<typename Op>
			static Tensor<T> ElementWiseBinaryOp(const Tensor<T>& lhs, const Tensor<T>& rhs, Op op)
			{
				Tensor<T> ret;

				ret.Resize(BroadcastShape(lhs.Shape(), rhs.Shape()));

				//StaticArray<int, 4> index;
				//index.ResizeZeroed(ret.Dim());
				//for (int i = 0; i < ret.LinearSize(); i++, ret.IterateIndex(index))
				parallel_for(0u, (uint)ret.LinearSize(), [&](int i)
				{
					TensorShape leftIndex;
					leftIndex.Resize(lhs.Dim());

					TensorShape rightIndex;
					rightIndex.Resize(rhs.Dim());

					TensorShape index = ret.Index(i);
					for (int j = 0; j < lhs.Dim(); j++)
					{
						leftIndex[j] = index[j + ret.Dim() - lhs.Dim()];
						if (leftIndex[j] >= lhs.Shape(j))
							leftIndex[j] = 0;
					}

					for (int j = 0; j < rhs.Dim(); j++)
					{
						rightIndex[j] = index[j + ret.Dim() - rhs.Dim()];
						if (rightIndex[j] >= rhs.Shape(j))
							rightIndex[j] = 0;
					}

					ret[i] = op(lhs(leftIndex), rhs(rightIndex));
				});

				return ret;
			}

			template<typename Op>
			static void ElementWiseBinaryOpInplace(Tensor<T>& lhs, const Tensor<T>& rhs, Op op)
			{
				auto newShape = BroadcastShape(lhs.Shape(), rhs.Shape());
				if (newShape != lhs.Shape())
				{
					lhs.Resize(newShape);
				}

#if __CUDACC__
				InvokeElementWiseBinaryOpInplace(lhs, rhs, op);
#endif

				//parallel_for(0u, (uint)lhs.LinearSize(), [&](int i)
				//{
				//	TensorShape leftIndex;
				//	leftIndex.Resize(lhs.Dim());

				//	TensorShape rightIndex;
				//	rightIndex.Resize(rhs.Dim());

				//	TensorShape index = lhs.Index(i);
				//	for (int j = 0; j < lhs.Dim(); j++)
				//	{
				//		leftIndex[j] = index[j + lhs.Dim() - lhs.Dim()];
				//		if (leftIndex[j] >= lhs.Shape(j))
				//			leftIndex[j] = 0;
				//	}

				//	for (int j = 0; j < rhs.Dim(); j++)
				//	{
				//		rightIndex[j] = index[j + lhs.Dim() - rhs.Dim()];
				//		if (rightIndex[j] >= rhs.Shape(j))
				//			rightIndex[j] = 0;
				//	}

				//	lhs[i] = op(lhs(leftIndex), rhs(rightIndex));
				//});
			}

			template<typename Op>
			static Tensor<T> ElementWiseUnaryOp(const Tensor<T>& lhs, Op op)
			{
				Tensor<T> ret;

				ret.Resize(lhs.Shape());
				for (int i = 0; i < ret.LinearSize(); i++)
				{
					ret[i] = op(lhs[i]);
				}

				return ret;
			}

			template<typename Op>
			static Tensor<T> ProjectionOp(const Tensor<T>& lhs, const TensorShape& axises, const bool keepDim, Op op, T initVal)
			{
				if (axises.Size() == 1 && axises[0] == -1)
				{
					//return Algorithm::Accumulate(lhs, initVal, op);

					thrust::reduce(lhs.Data(), lhs.Data() + lhs.LinearSize(), initVal, op);
					return lhs;
				}

				const TensorShape& inShape = lhs.Shape();
				TensorShape projShape;
				for (int i = 0; i < inShape.Size(); i++)
				{
					if (!keepDim)
					{
						if (!axises.Contains(i))
							projShape.Add(inShape[i]);
					}
					else
					{
						if (axises.Contains(i))
							projShape.Add(1);
						else
							projShape.Add(inShape[i]);
					}
				}

				if (projShape.Empty())
					projShape.Add(1);

				Tensor<T> ret;
				ret.Assign(initVal, TensorShape(projShape));

				TensorShape index;
				index.ResizeZeroed(lhs.Dim());
				for (int i = 0; i < lhs.LinearSize(); i++, lhs.IterateIndex(index))
				{
					TensorShape projIndex;

					for (int j = 0; j < index.Size(); j++)
					{
						if (!keepDim)
						{
							if (!axises.Contains(j))
								projIndex.Add(index[j]);
						}
						else
						{
							if (axises.Contains(j))
								projIndex.Add(0);
							else
								projIndex.Add(index[j]);
						}
					}

					if (projIndex.Empty())
						projIndex.Add(0);

					ret(projIndex) = op(ret(projIndex), lhs[i]);
				}

				return ret;
			}

	private:

			// --------------------------------------------------------------------------------------------
			// DO NOT USE DIRECTLY
			// STL-like iterators to enable range-based for loop support.
			// --------------------------------------------------------------------------------------------
			__forceinline friend		T*	begin(Tensor<T>& tensor) { return tensor.Data(); }
			__forceinline friend const	T*	begin(const Tensor<T>& tensor) { return tensor.Data(); }
			__forceinline friend		T*	end(Tensor<T>& tensor) { return tensor.Data() + tensor.LinearSize(); }
			__forceinline friend const	T*	end(const Tensor<T>& tensor) { return tensor.Data() + tensor.LinearSize(); }
		};

		using Tensorf = Tensor<float>;
		using Tensord = Tensor<double>;
		using Tensori = Tensor<int>;

		using Label = float;
		using Labels = Tensor<Label>;
	}
}