#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Containers/Array.h"
#include "Containers/Algorithm.h"
#include "Math/SparseMatrix.h"
#include "Core/Random.h"

#include "../OpenBLAS/include/cblas.h"
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "../OpenBLAS/include/lapacke.h"

#include <iostream>

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

			TENSOR_INLINE bool IterateIndex(TensorShape& index, const TensorShape& axises /*Axises to iterate through*/) const
			{
				for (int i = axises.Size() - 1; i >= 0; i--)
				{
					int axis = axises[i];

					index[axis]++;

					if (index[axis] < mShape[axis])
					{
						break;
					}
					else
					{
						if (i == 0)
							return false;

						index[axis] = 0;
					}
				}

				return true;
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


		template<typename ExpType>
		struct TExp
		{
			__forceinline const ExpType& Self() const
			{
				return *static_cast<const ExpType*>(this);
			}
		};
		
#ifdef __CUDACC__
		#include "TensorKernels.cuh"
#endif

		// Singleton for storing cublas handle
		class Cublas
		{
		public:
			static cublasHandle_t& GetHandle()
			{
				static cublasHandle_t Handle;
				return Handle;
			}
		public:
			Cublas(Cublas const&) = delete;
			void operator = (Cublas const&) = delete;
		};

		enum DeviceType
		{
			CPU, GPU
		};

		template<class T, DeviceType TDeviceType = GPU>
		class Tensor : public TExp<Tensor<T, TDeviceType>>
		{
		protected:
			T* mpData;
			TensorParams mParams;
			bool mbDataOwner = true;

		public:

			DeviceType GetDeviceType() const
			{
				return TDeviceType;
			}

			bool IsDataOwner() const
			{
				return mbDataOwner;
			}

			Tensor()
				: mpData(nullptr)
				, mbDataOwner(true)
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

			template<DeviceType TDeviceType2>
			Tensor(const Tensor<T, TDeviceType2>& rhs)
				: mpData(nullptr)
			{
				Resize(rhs.Shape());

				if (TDeviceType == CPU && rhs.GetDeviceType() == GPU)
					cudaMemcpy(mpData, rhs.Data(), LinearSize() * sizeof(T), cudaMemcpyDeviceToHost);
				else if (TDeviceType == GPU && rhs.GetDeviceType() == CPU)
					cudaMemcpy(mpData, rhs.Data(), LinearSize() * sizeof(T), cudaMemcpyHostToDevice);

				mbDataOwner = rhs.IsDataOwner();
			}

			Tensor& operator = (const Tensor& rhs)
			{
				Resize(rhs.Shape());

				if (TDeviceType == CPU)
					Memory::Memcpy(mpData, rhs.mpData, LinearSize() * sizeof(T));
				else if (TDeviceType == GPU)
					cudaMemcpy(mpData, rhs.mpData, LinearSize() * sizeof(T), cudaMemcpyDeviceToDevice);

				mbDataOwner = rhs.mbDataOwner;
				return *this;
			}

			//template<DeviceType TDeviceType2>
			//void Assign(const Tensor<T, TDeviceType2>& rhs)
			//{
			//	Resize(rhs.Shape());

			//	if (TDeviceType == CPU && rhs.GetDeviceType() == GPU)
			//		cudaMemcpy(mpData, rhs.Data(), LinearSize() * sizeof(T), cudaMemcpyDeviceToHost);
			//	else if (TDeviceType == GPU && rhs.GetDeviceType() == CPU)
			//		cudaMemcpy(mpData, rhs.Data(), LinearSize() * sizeof(T), cudaMemcpyHostToDevice);

			//	mReleaseData = rhs.GetReleaseData();
			//}

			Tensor& operator = (Tensor&& rhs)
			{
				Free();
				mParams = rhs.mParams;
				mpData = rhs.mpData;
				mbDataOwner = rhs.mbDataOwner;
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
				if (TDeviceType == CPU)
					*mpData = val;
				else if (TDeviceType == GPU)
					cudaMemcpy(mpData, &val, sizeof(T), cudaMemcpyHostToDevice);
				mbDataOwner = true;
				return *this;
			}

			Tensor<T, TDeviceType> Self() const
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
			inline Tensor<T, TDeviceType>& operator = (const TExp<EType>& rhs)
			{
				const EType& src = rhs.Self();
				Resize(src.Shape());

				if (TDeviceType == CPU)
				{
					parallel_for(0u, (uint)LinearSize(), [&](int i)
					{
						mpData[i] = src.Eval(i, mParams);
					});
				}
				else if (TDeviceType == GPU)
				{
#ifdef __CUDACC__
					InvokeExecuteExpression(src, mpData, mParams);
#endif
				}

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

				if (TDeviceType == CPU)
				{
					InitListNestedCopy(Data(), initList);
				}
				else if (TDeviceType == GPU)
				{
					T* pTempData = new T[LinearSize()];
					T* pIter = pTempData;
					InitListNestedCopy(pIter, initList);

					cudaMemcpy(Data(), pTempData, LinearSize() * sizeof(T), cudaMemcpyHostToDevice);

					Memory::SafeDeleteArray(pTempData);
				}
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

					if (TDeviceType == CPU)
						mpData = Memory::AlignedAlloc<T>(mParams.LinearSize());
					else if (TDeviceType == GPU)
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

				if (TDeviceType == CPU)
					Memory::Memcpy(mpData, pData, LinearSize() * sizeof(T));
				else if (TDeviceType == GPU)
					cudaMemcpy(mpData, pData, LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
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

			void MoveDevicePtr(T* pData, const TensorShape& shape)
			{
				Resize(shape);
				mpData = pData;
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
				ret.mbDataOwner = false;

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
				ret.mbDataOwner = false;
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
				ret.mbDataOwner = false;
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
				ret.mbDataOwner = false;
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

				ret.mpData = (T*)&this->operator()(filledIndex);
				ret.mParams = mParams.GetSliceIndex(index.Size());
				ret.mbDataOwner = false;
				return ret;
			}

			Tensor GetSection(const int from, const int to)
			{
				Assert(from < to);

				Tensor ret;

				ret.mpData = mpData + from * mParams.Stride(0);
				ret.mParams = mParams.GetSectionIndex(to - from);
				ret.mbDataOwner = false;

				return ret;
			}

			const Tensor GetSection(const int from, const int to) const
			{
				Assert(from < to);

				Tensor ret;

				ret.mpData = mpData + from * mParams.Stride(0);
				ret.mParams = mParams.GetSectionIndex(to - from);
				ret.mbDataOwner = false;

				return ret;
			}

			__forceinline bool IsTransposed() const
			{
				return mParams.IsTransposed();
			}

			__forceinline void Clear()
			{
				if (TDeviceType == CPU)
				{
					Memory::SafeClear(mpData, mParams.LinearSize());
				}
				else if (TDeviceType == GPU)
				{
					if (mpData)
						cudaMemset(mpData, 0, mParams.LinearSize() * sizeof(T));
				}

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
			TENSOR_INLINE bool IterateIndex(TensorShape& index, const TensorShape& axises) const
			{
				return mParams.IterateIndex(index, axises);
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

			void Set(const TensorShape& idx, const T val)
			{
				if (TDeviceType == CPU)
				{
					mpData[LinearIndex(idx)] = val;
				}
				else if (TDeviceType == GPU)
				{
					cudaMemcpy(&mpData[LinearIndex(idx)], &val, sizeof(T), cudaMemcpyHostToDevice);
				}
			}
			const T Get(const TensorShape& idx) const
			{
				if (TDeviceType == CPU)
				{
					return mpData[LinearIndex(idx)];
				}
				else if (TDeviceType == GPU)
				{
					T ret;
					cudaMemcpy(&ret, &mpData[LinearIndex(idx)], sizeof(T), cudaMemcpyDeviceToHost);
					return ret;
				}

				AssertNoEntry();
				return 0;
			}
			void Set(const int idx, const T val)
			{
				Assert(idx < mParams.LinearSize());
				if (TDeviceType == CPU)
				{
					mpData[idx] = val;
				}
				else if (TDeviceType == GPU)
				{
					cudaMemcpy(&mpData[idx], &val, sizeof(T), cudaMemcpyHostToDevice);
				}
			}
			const T Get(const int idx) const
			{
				Assert(idx < mParams.LinearSize());
				if (TDeviceType == CPU)
				{
					return mpData[idx];
				}
				else if (TDeviceType == GPU)
				{
					T ret;
					cudaMemcpy(&ret, &mpData[idx], sizeof(T), cudaMemcpyDeviceToHost);
					return ret;
				}

				AssertNoEntry();
				return 0;
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
				if (mbDataOwner)
				{
					if (TDeviceType == CPU)
					{
						Memory::SafeFree(mpData);
					}
					else if (TDeviceType == GPU)
					{
						if (mpData)
							cudaFree(mpData);
					}
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

				if (TDeviceType == CPU)
				{
					stream.ByteOrderWrite(A.Data(), A.LinearSize() * sizeof(T));
				}
				else if (TDeviceType == GPU)
				{
					const auto numSamples = A.LinearSize();
					Array<T> arr;
					arr.Resize(numSamples);
					cudaMemcpy(arr.Data(), A.Data(), numSamples * sizeof(T), cudaMemcpyDeviceToHost);

					stream.ByteOrderWrite(arr.Data(), arr.Size() * sizeof(T));
				}

				return stream;
			}

			friend Stream& operator >> (Stream& stream, Tensor& A)
			{
				// Load array.
				TensorShape newShape;
				stream >> newShape;

				A.Resize(newShape);

				if (TDeviceType == CPU)
				{
					stream.ByteOrderRead(A.Data(), A.LinearSize() * sizeof(T));
				}
				else if (TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(A.LinearSize());

					stream.ByteOrderRead(arr.Data(), arr.Size() * sizeof(T));

					cudaMemcpy(A.Data(), arr.Data(), A.LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
				}

				return stream;
			}

			friend std::ostream& operator << (std::ostream& stream, Tensor& A)
			{
				if (TDeviceType == CPU)
				{
					for (auto it : A)
						stream << it << " ";
				}
				else if (TDeviceType == GPU)
				{
					const auto numSamples = A.LinearSize();
					Array<T> arr;
					arr.Resize(numSamples);
					cudaMemcpy(arr.Data(), A.Data(), numSamples * sizeof(T), cudaMemcpyDeviceToHost);

					for (auto it : arr)
						stream << it << " ";
				}

				return stream;
			}

		public:
			// ----------------------------------------------------------------
			// Common utilities for tensors
			// ----------------------------------------------------------------
			static Tensor<T, TDeviceType> LinSpace(const T& start, const T& stop, const int& numSamples)
			{
				Tensor<T, TDeviceType> ret;
				ret.Resize(numSamples);

				T step = (stop - start) / T(numSamples - 1);

				if (TDeviceType == CPU)
				{
					for (int i = 0; i < numSamples; i++)
						ret[i] = start + i * step;
				}
				else if (TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(numSamples);

					for (int i = 0; i < numSamples; i++)
						arr[i] = start + i * step;

					cudaMemcpy(ret.Data(), arr.Data(), numSamples * sizeof(T), cudaMemcpyHostToDevice);
				}

				return ret;
			}

			static Tensor<T, TDeviceType> ArrayRange(const T& start, const T& stop, const T& step = 1)
			{
				Tensor<T, TDeviceType> ret;

				if (stop <= start)
					return ret;

				int numSamples = (stop - start) / step;
				ret.Resize(numSamples);

				if (TDeviceType == CPU)
				{
					for (int i = 0; i < numSamples; i++)
						ret[i] = start + i * step;
				}
				else if (TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(numSamples);

					for (int i = 0; i < numSamples; i++)
						arr[i] = start + i * step;

					cudaMemcpy(ret.Data(), arr.Data(), numSamples * sizeof(T), cudaMemcpyHostToDevice);
				}

				return ret;
			}

			static Tensor<T, TDeviceType> ArrayRange(const T& stop)
			{
				T start = 0;
				return ArrayRange(0, stop, 1);
			}

			static Tensor<T, TDeviceType> Identity(const int N)
			{
				Tensor<T, TDeviceType> ret;
				ret.Resize(N, N);

				if (TDeviceType == CPU)
				{
					for (int i = 0; i < N; i++)
						ret(i, i) = T(1);
				}
				else if (TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(N * N);

					for (int i = 0; i < N; i++)
						arr[ret.LinearIndex(i, i)] = T(1);

					cudaMemcpy(ret.Data(), arr.Data(), N * N * sizeof(T), cudaMemcpyHostToDevice);
				}

				return ret;
			}

			static Tensor<T, TDeviceType> Normalize(const Tensor<T, TDeviceType>& inTensor)
			{
				const Tensor<T, TDeviceType> tensorSqr = inTensor * inTensor;
				Tensor<T, TDeviceType> denorm = Tensor<T, TDeviceType>::Sqrt(Tensor<T, TDeviceType>::Sum(tensorSqr));

				return inTensor / denorm;
			}

			static Tensor<T, TDeviceType> Dot(const Tensor<T, TDeviceType>& lhs, const Tensor<T, TDeviceType>& rhs)
			{
				const TensorShape& leftShape = lhs.Shape();
				const TensorShape& rightShape = rhs.Shape();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

				Tensor<T, TDeviceType> ret;
				DotInplace(lhs, rhs, &ret);

				return ret;
			}

			static void DotInplace(const Tensor<T, TDeviceType>& lhs, const Tensor<T, TDeviceType>& rhs, Tensor<T, TDeviceType>* pResult, const float alpha = 1.0f, const float beta = 0.0f)
			{
				const TensorShape& leftShape = lhs.Shape();
				const TensorShape& rightShape = rhs.Shape();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

				pResult->Resize(leftShape[0], rightShape[1]);

				if (TDeviceType == CPU)
				{
					cblas_sgemm(CblasRowMajor,
						lhs.IsTransposed() ? CblasTrans : CblasNoTrans,
						rhs.IsTransposed() ? CblasTrans : CblasNoTrans,
						leftShape[0],
						rightShape[1],
						leftShape[1],
						alpha,
						lhs.Data(),
						lhs.IsTransposed() ? lhs.Shape(0) : lhs.Shape(1),
						rhs.Data(),
						rhs.IsTransposed() ? rhs.Shape(0) : rhs.Shape(1),
						beta,
						pResult->Data(),
						pResult->Shape(1));
				}
				else if (TDeviceType == GPU)
				{
					cublasSgemm(Cublas::GetHandle(),
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
			}
			
			static Tensor<T, TDeviceType> Dot(const SparseMatrix<T>& lhs, const Tensor<T, TDeviceType>& rhs)
			{
				const TensorShape& rightShape = rhs.Shape();

				Assertf(rightShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");
				Assertf(!(rightShape.Size() == 2 && lhs.n != rightShape[0]), "Dimension mismatch for tensor multiply.");

				Tensor<T, TDeviceType> ret;
				ret.Resize(lhs.n, rightShape[1]);

				DotInplace(lhs, rhs, &ret);

				return ret;
			}

			static void DotInplace(const SparseMatrix<T>& lhs, const Tensor<T, TDeviceType>& rhs, Tensor<T, TDeviceType>* pResult)
			{
				const TensorShape& rightShape = rhs.Shape();

				Assertf(rightShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");
				Assertf(!(rightShape.Size() == 2 && lhs.n != rightShape[0]), "Dimension mismatch for tensor multiply.");

				if (TDeviceType == CPU)
				{
					parallel_for(0, (int)lhs.n, [&](int i)
					{
						pResult[i] = 0;
						for (int j = lhs.rowstart[i]; j < lhs.rowstart[i + 1]; ++j)
						{
							pResult[i] += lhs.value[j] * rhs[lhs.colindex[j]];
						}
					});
				}
				else if (TDeviceType == GPU)
				{
					AssertNoEntry();
				}
			}

			static Tensor<T, TDeviceType> Transpose(const Tensor<T, TDeviceType>& inTensor, const TensorShape& transposeDim = {})
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

				Tensor<T, TDeviceType> ret;
				ret.Resize(transposedShape);

				TensorShape index;
				index.ResizeZeroed(inTensor.Dim());

				if (TDeviceType == CPU)
				{
					for (int i = 0; i < inTensor.LinearSize(); i++, inTensor.IterateIndex(index))
					{
						TensorShape transposedIndex = index;

						if (transposeDim.Empty())
						{
							Algorithm::Reverse(transposedIndex);
						}
						else
						{
							for (int j = 0; j < transposeDim.Size(); j++)
							{
								transposedIndex[j] = index[transposeDim[j]];
							}
						}

						ret(transposedIndex) = inTensor(index);
					}
				}
				else if (TDeviceType == GPU)
				{
					// TODO: Add CUDA kernel for matrix transpose
					AssertNoEntry();
				}

				return ret;
			}

			static Tensor<T, TDeviceType> Inverse(const Tensor<T, TDeviceType>& inTensor)
			{
				Assertf(inTensor.Dim() == 2 && inTensor.Shape(0) == inTensor.Shape(1),
					"Matrix inversion dimension mismatch.");

				const int N = inTensor.Shape(0);

				Tensor<T, TDeviceType> ret = inTensor;

				if (TDeviceType == CPU)
				{
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
				}
				else if (TDeviceType == GPU)
				{
					// TODO: Add CUDA kernel for matrix inversion
					AssertNoEntry();
				}

				return ret;
			}

			static Tensor<T, TDeviceType> Sum(const Tensor<T, TDeviceType>& inTensor, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Plus<>(), T(0));
			}

			static Tensor<T, TDeviceType> Product(const Tensor<T, TDeviceType>& inTensor, const TensorShape& axises = { -1 })
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Multiply<>(), T(1));
			}

			static Tensor<T, TDeviceType> Max(const Tensor<T, TDeviceType>& inTensor, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Max<>(), T(Math::EDX_NEG_INFINITY));
			}


			static void SumInplace(const Tensor<T, TDeviceType>& inTensor, Tensor<T, TDeviceType>* pResult, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				ProjectionOpInplace(inTensor, axises, keepDim, Algorithm::Plus<>(), T(0), pResult);
			}

			static void ProductInplace(const Tensor<T, TDeviceType>& inTensor, Tensor<T, TDeviceType>* pResult, const TensorShape& axises = { -1 })
			{
				ProjectionOpInplace(inTensor, axises, keepDim, Algorithm::Multiply<>(), T(1), pResult);
			}

			static void MaxInplace(const Tensor<T, TDeviceType>& inTensor, Tensor<T, TDeviceType>* pResult, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				ProjectionOpInplace(inTensor, axises, keepDim, Algorithm::Max<>(), T(Math::EDX_NEG_INFINITY), pResult);
			}

			static Tensor<T, TDeviceType> Mean(const Tensor<T, TDeviceType>& X, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				Tensor<T, TDeviceType> ret = Sum(X, axises);

				float invDivisor = ret.LinearSize() / float(X.LinearSize());
				ret *= invDivisor;

				return ret;
			}

			static Tensor<T, TDeviceType> StandardDeviation(const Tensor<T, TDeviceType>& X, const TensorShape& axises = { -1 }, const bool keepDim = false)
			{
				Tensor<T, TDeviceType> mean = Mean(X, axises, keepDim);
				Tensor<T, TDeviceType> centeredX = X - mean;

				Tensor<T, TDeviceType> variance = Tensorf::Mean(centeredX * centeredX, axises, keepDim);

				return TensorExpr::Sqrt(variance + Scalar(1e-5f));
			}

			template<typename... Shape>
			static Tensor<T, TDeviceType> RandomInt(const int high, Shape... shape)
			{
				Tensor<T, TDeviceType> ret;
				ret.Resize(shape...);

				RandomGen random;
				if (TDeviceType == CPU)
				{
					for (auto& it : ret)
					{
						it = random.UnsignedInt() % high;
					}
				}
				else if (TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(ret.LinearSize());
					for (auto& it : arr)
					{
						it = random.UnsignedInt() % high;
					}

					cudaMemcpy(ret.Data(), arr.Data(), ret.LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
				}

				return ret;
			}

			template<typename... Shape>
			static Tensor<T, TDeviceType> RandomFloat(Shape... shape)
			{
				Tensor<T, TDeviceType> ret;
				ret.Resize(shape...);

				RandomGen random;
				if (TDeviceType == CPU)
				{
					for (auto& it : ret)
					{
						it = random.Float();
					}
				}
				else if (TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(ret.LinearSize());
					for (auto& it : arr)
					{
						it = random.Float();
					}

					cudaMemcpy(ret.Data(), arr.Data(), ret.LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
				}

				return ret;
			}

			template<typename... Shape>
			static Tensor<T, TDeviceType> RandomNormalDistribution(const float std, Shape... shape)
			{
				Tensor<T, TDeviceType> ret;
				ret.Resize(shape...);

				RandomGen random;

				if (TDeviceType == CPU)
				{
					for (auto& it : ret)
					{
						it = random.GaussFloat(0.0f, std);
					}
				}
				else if(TDeviceType == GPU)
				{
					Array<T> arr;
					arr.Resize(ret.LinearSize());
					for (auto& it : arr)
					{
						it = random.GaussFloat(0.0f, std);
					}

					cudaMemcpy(ret.Data(), arr.Data(), ret.LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
				}

				return ret;
			}

			__forceinline void operator += (const Tensor<T, TDeviceType>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Plus<>());
			}
			__forceinline void operator -= (const Tensor<T, TDeviceType>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Substract<>());
			}

			__forceinline void operator *= (const Tensor<T, TDeviceType>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Multiply<>());
			}
			__forceinline void operator /= (const Tensor<T, TDeviceType>& rhs)
			{
				ElementWiseBinaryOpInplace(*this, rhs, Algorithm::Divide<>());
			}

		private:

			template<typename Op>
			static void ElementWiseBinaryOpInplace(Tensor<T, TDeviceType>& lhs, const Tensor<T, TDeviceType>& rhs, Op op)
			{
				auto newShape = BroadcastShape(lhs.Shape(), rhs.Shape());
				if (newShape != lhs.Shape())
				{
					lhs.Resize(newShape);
				}

				if (TDeviceType == CPU)
				{
					parallel_for(0u, (uint)lhs.LinearSize(), [&](int i)
					{
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
					});
				}
				else if (TDeviceType == GPU)
				{
#ifdef __CUDACC__
					InvokeElementWiseBinaryOpInplace(lhs, rhs, op);
#endif
				}
			}

			template<typename Op>
			static Tensor<T, TDeviceType> ProjectionOp(const Tensor<T, TDeviceType>& lhs, const TensorShape& axises, const bool keepDim, Op op, T initVal)
			{
				if (axises.Size() == 1 && axises[0] == -1)
				{
					if (TDeviceType == CPU)
					{
						return Algorithm::Accumulate(lhs, initVal, op);
					}
					else if (TDeviceType == GPU)
					{
#ifdef __CUDACC__
						T* reduced = InvokeReduce(lhs.Data(), lhs.LinearSize(), initVal, op);
						Tensor<T, TDeviceType> ret;
						ret.MoveDevicePtr(reduced, { 1 });
						return ret;
#else
						AssertNoEntry();
						return 0.0f;
#endif
					}
				}

				const TensorShape& inShape = lhs.Shape();
				TensorShape projShape;
				TensorShape projShapeKeepDim;
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

					if (axises.Contains(i))
						projShapeKeepDim.Add(1);
					else
						projShapeKeepDim.Add(inShape[i]);
				}

				if (projShape.Empty())
					projShape.Add(1);

				if (projShapeKeepDim.Empty())
					projShapeKeepDim.Add(1);

				TensorParams tensorParamsKeepDim;
				tensorParamsKeepDim.Resize(projShapeKeepDim);

				Tensor<T, TDeviceType> ret;
				if (TDeviceType == CPU)
				{
					ret.Assign(initVal, TensorShape(projShape));

					parallel_for(0, (int)tensorParamsKeepDim.LinearSize(), [&](int i)
					{
						TensorShape projIndex = tensorParamsKeepDim.Index(i);

						do
						{
							ret[i] = op(ret[i], lhs(projIndex));
						}
						while (lhs.IterateIndex(projIndex, axises));
					});
				}
				else if (TDeviceType == GPU)
				{
					ret.Resize(projShape);
#ifdef __CUDACC__
					InvokeTensorProjectionOp(ret, lhs, tensorParamsKeepDim, axises, op, initVal);
#endif
				}

				return ret;
			}

			template<typename Op>
			static void ProjectionOpInplace(const Tensor<T, TDeviceType>& lhs, const TensorShape& axises, const bool keepDim, Op op, T initVal, Tensor<T, TDeviceType>* pResult)
			{
				if (axises.Size() == 1 && axises[0] == -1)
				{
					if (TDeviceType == CPU)
					{
						*pResult = { Algorithm::Accumulate(lhs, initVal, op) };
					}
					else if (TDeviceType == GPU)
					{
#ifdef __CUDACC__
						T* reduced = InvokeReduce(lhs.Data(), lhs.LinearSize(), initVal, op);
						pResult->MoveDevicePtr(reduced, { 1 });
#else
						AssertNoEntry();
#endif
					}
				}

				const TensorShape& inShape = lhs.Shape();
				TensorShape projShape;
				TensorShape projShapeKeepDim;
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

					if (axises.Contains(i))
						projShapeKeepDim.Add(1);
					else
						projShapeKeepDim.Add(inShape[i]);
				}

				if (projShape.Empty())
					projShape.Add(1);

				if (projShapeKeepDim.Empty())
					projShapeKeepDim.Add(1);

				TensorParams tensorParamsKeepDim;
				tensorParamsKeepDim.Resize(projShapeKeepDim);

				if (TDeviceType == CPU)
				{
					pResult->Assign(initVal, TensorShape(projShape));

					parallel_for(0, (int)tensorParamsKeepDim.LinearSize(), [&](int i)
					{
						TensorShape projIndex = tensorParamsKeepDim.Index(i);

						do
						{
							(*pResult)[i] = op((*pResult)[i], lhs(projIndex));
						} while (lhs.IterateIndex(projIndex, axises));
					});
				}
				else if (TDeviceType == GPU)
				{
					pResult->Resize(projShape);
#ifdef __CUDACC__
					InvokeTensorProjectionOp(*pResult, lhs, tensorParamsKeepDim, axises, op, initVal);
#endif
				}
			}

	private:

			// --------------------------------------------------------------------------------------------
			// DO NOT USE DIRECTLY
			// STL-like iterators to enable range-based for loop support.
			// --------------------------------------------------------------------------------------------
			__forceinline friend		T*	begin(Tensor<T, TDeviceType>& tensor) { return tensor.Data(); }
			__forceinline friend const	T*	begin(const Tensor<T, TDeviceType>& tensor) { return tensor.Data(); }
			__forceinline friend		T*	end(Tensor<T, TDeviceType>& tensor) { return tensor.Data() + tensor.LinearSize(); }
			__forceinline friend const	T*	end(const Tensor<T, TDeviceType>& tensor) { return tensor.Data() + tensor.LinearSize(); }
		};

		using Tensorf = Tensor<float>;
		using Tensord = Tensor<double>;
		using Tensori = Tensor<int>;
		
		#include "TemplateExpression.h"
	}
}