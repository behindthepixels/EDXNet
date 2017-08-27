#pragma once

#include "Containers/Array.h"
#include "Containers/Algorithm.h"
#include "Math/SparseMatrix.h"

#include "../OpenBLAS/include/cblas.h"
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "../OpenBLAS/include/lapacke.h"

#include <ppl.h>
using namespace concurrency;

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

		class TensorIndex
		{
		protected:
			Array<int> mShape;
			Array<int> mStrides;
			int mLinearSize;
			bool mTransposed = false;

		public:
			TensorIndex()
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

			void Resize(const Array<int>& shape)
			{
				mShape = shape;
				Resize_Common();
			}

			template<int N>
			void Resize(const StaticArray<int, N>& shape)
			{
				mShape = shape;
				Resize_Common();
			}

			template<typename... Shape>
			void Reshape(Shape... shape)
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");

				auto oldSize = mLinearSize;

				mShape = { shape... };
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}

			void Reshape(const Array<int>& shape)
			{
				auto oldSize = mLinearSize;

				mShape = shape;
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}

			template<int N>
			void Reshape(const StaticArray<int, N>& shape)
			{
				auto oldSize = mLinearSize;

				mShape = shape;
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}
			
			void Transpose(const Array<int>& transposeDim = {})
			{
				if (transposeDim.Empty())
				{
					Algorithm::Reverse(mShape);
					Algorithm::Reverse(mStrides);
				}
				else
				{
					Array<int> shapeCopy = mShape;
					Array<int> strideCopy = mStrides;
					for (int i = 0; i < transposeDim.Size(); i++)
					{
						mShape[i] = shapeCopy[transposeDim[i]];
						mStrides[i] = strideCopy[transposeDim[i]];
					}
				}

				mTransposed = true;
			}

			template<typename... Index>
			__forceinline int LinearIndex(Index... idx) const
			{
				constexpr int size = sizeof...(idx);
				Assertf(size == mShape.Size(), "Input index dimension does not match with array dimension.");

				int _idx[size] = { idx... };
				return LinearIndex<size>(_idx);
			}

			__forceinline int LinearIndex(const Array<int>& idx) const
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
			__forceinline int LinearIndex(const int idx[N]) const
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

			template<int N>
			__forceinline int LinearIndex(const StaticArray<int, N>& idx) const
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

			__forceinline Array<int> Index(int linearIdx) const
			{
				Assert(linearIdx < mLinearSize);

				Array<int> vRet;
				vRet.Resize(mShape.Size());
				for (int i = 0; i < mShape.Size(); i++)
				{
					vRet[i] = linearIdx / mStrides[i];
					linearIdx %= mStrides[i];
				}
				return vRet;
			}

			__forceinline StaticArray<int, 4> StaticIndex(int linearIdx) const
			{
				Assert(linearIdx < mLinearSize);

				StaticArray<int, 4> vRet;
				vRet.Resize(mShape.Size());
				for (int i = 0; i < mShape.Size(); i++)
				{
					vRet[i] = linearIdx / mStrides[i];
					linearIdx %= mStrides[i];
				}
				return vRet;
			}

			__forceinline int LinearSize() const
			{
				return mLinearSize;
			}
			
			__forceinline bool IndexRangeCheck(const Array<int>& index) const
			{
				for (int i = 0; i < mShape.Size(); i++)
				{
					if (index[i] >= mShape[i])
						return false;
				}
				return true;
			}

			__forceinline void IterateIndex(Array<int>& index) const
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

			__forceinline bool IndexRangeCheck(const StaticArray<int, 4>& index) const
			{
				for (int i = 0; i < mShape.Size(); i++)
				{
					if (index[i] >= mShape[i])
						return false;
				}
				return true;
			}

			__forceinline void IterateIndex(StaticArray<int, 4>& index) const
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

			__forceinline int Shape(int iDim) const
			{
				Assert(iDim < mShape.Size());
				return mShape[iDim];
			}

			__forceinline const Array<int>& Shape() const
			{
				return mShape;
			}

			__forceinline Array<int> Shape()
			{
				return mShape;
			}

			__forceinline int Stride(int iDim) const
			{
				Assert(iDim < mShape.Size());
				return mStrides[iDim];
			}

			__forceinline TensorIndex GetSliceIndex(int subDim) const
			{
				Assert(subDim <= mShape.Size());
				TensorIndex ret;

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

			__forceinline TensorIndex GetSectionIndex(int num) const
			{
				Assert(num < mShape[0]);
				TensorIndex ret;

				ret.mShape = mShape;
				ret.mShape[0] = num;

				ret.Resize_Common();
				return ret;
			}

			__forceinline bool IsTransposed() const
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

		static Array<int> BroadcastShape(const Array<int>& leftShape, const Array<int>& rightShape)
		{
			if (leftShape == rightShape) // Trivially broadcasted
			{
				return leftShape;
			}

			const int leftDim = leftShape.Size();
			const int rightDim = rightShape.Size();
			const auto& greaterShape = leftDim > rightDim ? leftShape : rightShape;
			const int retDim = greaterShape.Size();

			Array<int> ret;
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

		template<class T>
		class Tensor : public TExp<Tensor<T>>
		{
		protected:
			T* mpData;
			TensorIndex mIndex;
			bool mReleaseData = true;

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
				Memory::Memcpy(mpData, rhs.mpData, LinearSize() * sizeof(T));
				mReleaseData = rhs.mReleaseData;
				return *this;
			}

			Tensor& operator = (Tensor&& rhs)
			{
				Free();
				mIndex = rhs.mIndex;
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

				parallel_for(0u, (uint)LinearSize(), [&](int i)
				{
					mpData[i] = src.Eval(i, mIndex);
				});

				return *this;
			}

			__forceinline T Eval(const int idx, const TensorIndex& broadcastIndex) const
			{
				StaticArray<int, 4> selfIndex;
				selfIndex.Resize(Dim());

				StaticArray<int, 4> index = broadcastIndex.StaticIndex(idx);
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
				Resize(DeriveShapeFromNestedInitList<Array<int>>(initList));
				InitListNestedCopy(Data(), initList);
				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 2> initList)
			{
				Resize(DeriveShapeFromNestedInitList<Array<int>>(initList));
				InitListNestedCopy(Data(), initList);
				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 3> initList)
			{
				Resize(DeriveShapeFromNestedInitList<Array<int>>(initList));
				InitListNestedCopy(Data(), initList);
				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 4> initList)
			{
				Resize(DeriveShapeFromNestedInitList<Array<int>>(initList));
				InitListNestedCopy(Data(), initList);
				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 5> initList)
			{
				Resize(DeriveShapeFromNestedInitList<Array<int>>(initList));
				InitListNestedCopy(Data(), initList);
				return *this;
			}

			template<typename... TShape>
			void Resize(TShape... shape)
			{
				constexpr int size = sizeof...(shape);
				Resize<size>({ shape... });
			}

			void Resize(const Array<int>& shape)
			{
				int newLinSize = Algorithm::Accumulate(shape, int(1), Algorithm::Multiply<>());
				if (LinearSize() != newLinSize)
				{
					Free();
					mIndex.Resize(shape);

					mpData = Memory::AlignedAlloc<T>(mIndex.LinearSize());
					Assert(mpData);

					Clear();
				}
				else if (Shape() != shape)
				{
					Reshape(shape);
				}
			}

			template<int N>
			void Resize(const StaticArray<int, N>& shape)
			{
				int newLinSize = Algorithm::Accumulate(shape, int(1), Algorithm::Multiply<>());
				if (LinearSize() != newLinSize)
				{
					Free();
					mIndex.Resize<N>(shape);

					mpData = Memory::AlignedAlloc<T>(mIndex.LinearSize());
					Assert(mpData);

					Clear();
				}
				else
				{
					const auto& oldShape = Shape();
					if (oldShape.Size() != N)
					{
						Reshape<N>(shape);
					}
					else
					{
						for (int i = 0; i < N; i++)
						{
							if (oldShape[i] != shape[i])
							{
								Reshape<N>(shape);
								break;
							}
						}
					}
				}
			}

			void Assign(const T* pData, const Array<int>& shape)
			{
				Resize(shape);
				Memory::Memcpy(mpData, pData, LinearSize() * sizeof(T));
			}

			void Assign(const T val, const Array<int>& shape)
			{
				Resize(shape);

				for (int i = 0; i < LinearSize(); i++)
					mpData[i] = val;
			}

			template<typename... Shape>
			Tensor Reshape(Shape... shape)
			{
				mIndex.Reshape(shape...);
				return *this;
			}

			Tensor Reshape(const Array<int>& shape)
			{
				mIndex.Reshape(shape);
				return *this;
			}

			template<int N>
			Tensor Reshape(const StaticArray<int, N>& shape)
			{
				mIndex.Reshape<N>(shape);
				return *this;
			}

			Tensor GetTransposed(const Array<int>& transposeDim = {})
			{
				Tensor ret;
				ret.mIndex = mIndex;
				ret.mpData = mpData;
				ret.mReleaseData = false;

				ret.mIndex.Transpose(transposeDim);

				return ret;
			}

			template<typename... Shape>
			Tensor GetWithShape(Shape... shape)
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");
				constexpr int N = sizeof...(shape);

				return GetWithShape<N>({ shape... });
			}

			Tensor GetWithShape(const Array<int>& shape)
			{
				Tensor ret;
				ret.mIndex = mIndex;
				ret.mIndex.Reshape(shape);
				ret.mpData = mpData;
				ret.mReleaseData = false;
				return ret;
			}

			template<int N>
			Tensor GetWithShape(const StaticArray<int, N>& shape)
			{
				Tensor ret;
				ret.mIndex = mIndex;
				ret.mIndex.Reshape<N>(shape);
				ret.mpData = mpData;
				ret.mReleaseData = false;
				return ret;
			}

			template<typename... Shape>
			const Tensor GetWithShape(Shape... shape) const
			{
				static_assert(AllIntegralType<Shape...>::Value, "All parameters have to be integral type.");
				constexpr int N = sizeof...(shape);

				return GetWithShape<N>({ shape... });
			}

			const Tensor GetWithShape(const Array<int>& shape) const
			{
				Tensor ret;
				ret.mIndex = mIndex;
				ret.mIndex.Reshape(shape);
				ret.mpData = mpData;
				ret.mReleaseData = false;
				return ret;
			}

			template<int N>
			const Tensor GetWithShape(const StaticArray<int, N>& shape) const
			{
				Tensor ret;
				ret.mIndex = mIndex;
				ret.mIndex.Reshape<N>(shape);
				ret.mpData = mpData;
				ret.mReleaseData = false;
				return ret;
			}

			template<typename... Index>
			Tensor GetSlice(Index... index)
			{
				static_assert(AllIntegralType<Index...>::Value, "All parameters have to be integral type.");
				constexpr int N = sizeof...(index);

				return GetSlice<N>({ index... });
			}

			Tensor GetSlice(const Array<int>& index)
			{
				Tensor ret;

				Array<int> filledIndex = index;
				int numComponentToFill = Dim() - index.Size();
				while (numComponentToFill--)
				{
					filledIndex.Add(0);
				}

				ret.mpData = &this->operator()(filledIndex);
				ret.mIndex = mIndex.GetSliceIndex(index.Size());
				ret.mReleaseData = false;
				return ret;
			}

			template<int N>
			Tensor GetSlice(const StaticArray<int, N>& index)
			{
				Tensor ret;

				Array<int> filledIndex = Array<int>(index);
				int numComponentToFill = Dim() - index.Size();
				while (numComponentToFill--)
				{
					filledIndex.Add(0);
				}

				ret.mpData = &this->operator()(filledIndex);
				ret.mIndex = mIndex.GetSliceIndex(index.Size());
				ret.mReleaseData = false;
				return ret;
			}


			template<typename... Index>
			const Tensor GetSlice(Index... index) const
			{
				static_assert(AllIntegralType<Index...>::Value, "All parameters have to be integral type.");
				constexpr int N = sizeof...(index);

				return GetSlice<N>({ index... });
			}

			const Tensor GetSlice(const Array<int>& index) const
			{
				Tensor ret;

				Array<int> filledIndex = index;
				int numComponentToFill = Dim() - index.Size();
				while (numComponentToFill--)
				{
					filledIndex.Add(0);
				}

				ret.mpData = &this->operator()(filledIndex);
				ret.mIndex = mIndex.GetSliceIndex(index.Size());
				ret.mReleaseData = false;
				return ret;
			}

			template<int N>
			const Tensor GetSlice(const StaticArray<int, N>& index) const
			{
				Tensor ret;

				Array<int> filledIndex = Array<int>(index);
				int numComponentToFill = Dim() - index.Size();
				while (numComponentToFill--)
				{
					filledIndex.Add(0);
				}

				ret.mpData = &this->operator()(filledIndex);
				ret.mIndex = mIndex.GetSliceIndex(index.Size());
				ret.mReleaseData = false;
				return ret;
			}

			Tensor GetSection(const int from, const int to)
			{
				Assert(from < to);

				Tensor ret;

				ret.mpData = mpData + from * mIndex.Stride(0);
				ret.mIndex = mIndex.GetSectionIndex(to - from);
				ret.mReleaseData = false;

				return ret;
			}

			const Tensor GetSection(const int from, const int to) const
			{
				Assert(from < to);

				Tensor ret;

				ret.mpData = mpData + from * mIndex.Stride(0);
				ret.mIndex = mIndex.GetSectionIndex(to - from);
				ret.mReleaseData = false;

				return ret;
			}

			__forceinline bool IsTransposed() const
			{
				return mIndex.IsTransposed();
			}

			__forceinline void Clear()
			{
				Memory::SafeClear(mpData, mIndex.LinearSize());
			}

			template<typename... Index>
			__forceinline int LinearIndex(Index... idx) const
			{
				return mIndex.LinearIndex(idx...);
			}
			__forceinline int LinearIndex(const Array<int>& idx) const
			{
				return mIndex.LinearIndex(idx);
			}
			__forceinline virtual Array<int> Index(int linearIdx) const
			{
				return mIndex.Index(linearIdx);
			}
			__forceinline virtual StaticArray<int, 4> StaticIndex(int linearIdx) const
			{
				return mIndex.StaticIndex(linearIdx);
			}
			__forceinline int LinearSize() const
			{
				return mIndex.LinearSize();
			}
			__forceinline bool IndexRangeCheck(const Array<int>& index) const
			{
				return mIndex.IndexRangeCheck(index);
			}
			__forceinline void IterateIndex(Array<int>& index) const
			{
				return mIndex.IterateIndex(index);
			}
			__forceinline bool IndexRangeCheck(const StaticArray<int, 4>& index) const
			{
				return mIndex.IndexRangeCheck(index);
			}
			__forceinline void IterateIndex(StaticArray<int, 4>& index) const
			{
				return mIndex.IterateIndex(index);
			}
			__forceinline int Shape(int iDim) const
			{
				return mIndex.Shape(iDim);
			}
			__forceinline const Array<int>& Shape() const
			{
				return mIndex.Shape();
			}
			__forceinline Array<int> Shape()
			{
				return mIndex.Shape();
			}
			__forceinline int Dim() const
			{
				return Shape().Size();
			}
			__forceinline int Stride(int iDim) const
			{
				return mIndex.Stride(iDim);
			}
			__forceinline bool Empty() const
			{
				return LinearSize() == 0;
			}

			template<typename... Index>
			__forceinline T& operator () (Index... idx)
			{
				return mpData[LinearIndex(idx...)];
			}

			template<typename... Index>
			__forceinline const T& operator () (Index... idx) const
			{
				return mpData[LinearIndex(idx...)];
			}

			__forceinline T& operator [] (const Array<int>& idx)
			{
				return mpData[LinearIndex(idx)];
			}
			__forceinline const T& operator [] (const Array<int>& idx) const
			{
				return mpData[LinearIndex(idx)];
			}
			__forceinline T& operator [] (const int idx)
			{
				Assert(idx < mIndex.LinearSize());
				return mpData[idx];
			}
			__forceinline const T& operator [] (const int idx) const
			{
				Assert(idx < mIndex.LinearSize());
				return mpData[idx];
			}
			__forceinline const T* Data() const
			{
				return mpData;
			}
			__forceinline T* Data()
			{
				return mpData;
			}

			void Free()
			{
				if (mReleaseData)
					Memory::SafeFree(mpData);
			}

			// ----------------------------------------------------------------
			// Serialization operator
			// ----------------------------------------------------------------
			friend Stream& operator << (Stream& stream, Tensor& A)
			{
				// Save array.
				Array<int> shape = A.Shape();
				stream << shape;
				stream.ByteOrderWrite(A.Data(), A.LinearSize() * sizeof(T));

				return stream;
			}

			friend Stream& operator >> (Stream& stream, Tensor& A)
			{
				// Load array.
				Array<int> newShape;
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
				const Array<int>& leftShape = lhs.Shape();
				const Array<int>& rightShape = rhs.Shape();

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
				const Array<int>& leftShape = lhs.Shape();
				const Array<int>& rightShape = rhs.Shape();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");


				static const uint BLOCK_SIZE = 90;

				cblas_sgemm(CblasRowMajor,
					lhs.IsTransposed() ? CblasTrans : CblasNoTrans,
					rhs.IsTransposed() ? CblasTrans : CblasNoTrans,
					leftShape[0],
					rightShape[1],
					leftShape[1],
					1.0f,
					lhs.Data(),
					lhs.IsTransposed() ? lhs.Shape(0) : lhs.Shape(1),
					rhs.Data(),
					rhs.IsTransposed() ? rhs.Shape(0) : rhs.Shape(1),
					0.0f,
					pResult->Data(),
					pResult->Shape(1));
			}
			
			static Tensor<T> Dot(const SparseMatrix<T>& lhs, const Tensor<T>& rhs)
			{
				const Array<int>& rightShape = rhs.Shape();

				Assertf(rightShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");
				Assertf(!(rightShape.Size() == 2 && lhs.n != rightShape[0]), "Dimension mismatch for tensor multiply.");

				Tensor<T> ret;
				ret.Resize(lhs.n, rightShape[1]);

				DotInplace(lhs, rhs, &ret);

				return ret;
			}

			static void DotInplace(const SparseMatrix<T>& lhs, const Tensor<T>& rhs, Tensor<T>* pResult)
			{
				const Array<int>& rightShape = rhs.Shape();

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

			static Tensor<T> Transpose(const Tensor<T>& inTensor, const Array<int>& transposeDim = {})
			{
				const Array<int>& inShape = inTensor.Shape();

				Array<int> transposedShape = inShape;
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

				StaticArray<int, 4> index;
				index.ResizeZeroed(inTensor.Dim());
				for (int i = 0; i < inTensor.LinearSize(); i++, inTensor.IterateIndex(index))
				{
					StaticArray<int, 4> transposedIndex = index;

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

				Tensorf ret = inTensor;
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

			static Tensor<T> Sum(const Tensor<T>& inTensor, const Array<int>& axises = { -1 }, const bool keepDim = false)
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Plus<>(), T(0));
			}

			static Tensor<T> Product(const Tensor<T>& inTensor, const Array<int>& axises = { -1 })
			{
				return ProjectionOp(inTensor, axises, keepDim, Algorithm::Multiply<>(), T(1));
			}

			static Tensor<T> Max(const Tensor<T>& inTensor, const Array<int>& axises = { -1 }, const bool keepDim = false)
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

			static Tensor<T> Mean(const Tensor<T>& X, const Array<int>& axises = { -1 }, const bool keepDim = false)
			{
				Tensor<T> ret = Sum(X, axises);

				float invDivisor = ret.LinearSize() / float(X.LinearSize());
				ret *= invDivisor;

				return ret;
			}

			static Tensor<T> StandardDeviation(const Tensor<T>& X, const Array<int>& axises = { -1 }, const bool keepDim = false)
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
					StaticArray<int, 4> leftIndex;
					leftIndex.Resize(lhs.Dim());

					StaticArray<int, 4> rightIndex;
					rightIndex.Resize(rhs.Dim());

					StaticArray<int, 4> index = ret.StaticIndex(i);
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


				//StaticArray<int, 4> index;
				//index.ResizeZeroed(lhs.Dim());
				//for (int i = 0; i < lhs.LinearSize(); i++, lhs.IterateIndex(index))
				parallel_for(0u, (uint)lhs.LinearSize(), [&](int i)
				{
					StaticArray<int, 4> leftIndex;
					leftIndex.Resize(lhs.Dim());

					StaticArray<int, 4> rightIndex;
					rightIndex.Resize(rhs.Dim());

					StaticArray<int, 4> index = lhs.StaticIndex(i);
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
			static Tensor<T> ProjectionOp(const Tensor<T>& lhs, const Array<int>& axises, const bool keepDim, Op op, T initVal)
			{
				if (axises.Size() == 1 && axises[0] == -1)
				{
					return Algorithm::Accumulate(lhs, initVal, op);
				}

				const Array<int>& inShape = lhs.Shape();
				StaticArray<int, 4> projShape;
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
				ret.Assign(initVal, Array<int>(projShape));

				StaticArray<int, 4> index;
				index.ResizeZeroed(lhs.Dim());
				for (int i = 0; i < lhs.LinearSize(); i++, lhs.IterateIndex(index))
				{
					StaticArray<int, 4> projIndex;

					for (int i = 0; i < index.Size(); i++)
					{
						if (!keepDim)
						{
							if (!axises.Contains(i))
								projIndex.Add(index[i]);
						}
						else
						{
							if (axises.Contains(i))
								projIndex.Add(0);
							else
								projIndex.Add(index[i]);
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
		__forceinline friend T*			begin(Tensor<T>& tensor) { return tensor.Data(); }
		__forceinline friend const T*	begin(const Tensor<T>& tensor) { return tensor.Data(); }
		__forceinline friend T*			end(Tensor<T>& tensor) { return tensor.Data() + tensor.LinearSize(); }
		__forceinline friend const T*	end(const Tensor<T>& tensor) { return tensor.Data() + tensor.LinearSize(); }
		};

		using Tensorf = Tensor<float>;
		using Tensord = Tensor<double>;
		using Tensori = Tensor<int>;

		using Label = float;
		using Labels = Tensor<Label>;
	}
}