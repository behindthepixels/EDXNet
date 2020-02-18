#pragma once

// Template expression definition
// This file should be included before Tensor class declaration in Tensor.h


template<typename T>
struct TScalarExp : public TExp<TScalarExp<T>>
{
	const T val;

	TScalarExp(const T& _val)
		: val(_val)
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE T Eval(const int i, const TensorParams& broadcastIndex) const
	{
		return val;
	}

	template<typename TGrad>
	void Backward(const TExp<TGrad>& inGrad) const
	{
	}

	__forceinline TensorShape Shape() const
	{
		return{ 1 };
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
	}

};

template<typename T>
TScalarExp<T> Scalar(const T& val)
{
	return TScalarExp<T>(val);
}


template<typename TOp, typename TLhs, typename TRhs>
struct TBinaryExp : public TExp<TBinaryExp<TOp, TLhs, TRhs>>
{
	const TLhs lhs;
	const TRhs rhs;

	TBinaryExp(const TLhs& _lhs, const TRhs& _rhs)
		: lhs(_lhs.Self())
		, rhs(_rhs.Self())
	{
	}

	TBinaryExp(const TBinaryExp& _rhs)
		: lhs(_rhs.lhs.Self())
		, rhs(_rhs.rhs.Self())
	{
	}

	TBinaryExp(const TBinaryExp&& _rhs)
		: lhs(Move(_rhs.lhs.Self()))
		, rhs(Move(_rhs.rhs.Self()))
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		float val = TOp::Exec(lhs.Eval(i, broadcastIndex), rhs.Eval(i, broadcastIndex));
		return val;
	}

	template<typename TGrad>
	void Backward(const TExp<TGrad>& inGrad) const
	{
		const TGrad& grad = inGrad.Self();
		Assert(grad.Shape() == Shape());

		TensorShape leftShape = lhs.Shape();
		auto leftGrad = Unbroadcast(TOp::Backward(rhs, grad), leftShape);

		TensorShape rightShape = rhs.Shape();
		auto rightGrad = Unbroadcast(TOp::Backward(lhs, grad), rightShape);

		lhs.Backward(leftGrad);
		rhs.Backward(rightGrad);

		return;
	}

	__forceinline TensorShape Shape() const
	{
		TensorShape shape = BroadcastShape(lhs.Shape(), rhs.Shape());

		return shape;
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
		lhs.Preprocess(bForceRecompute);
		rhs.Preprocess(bForceRecompute);
		return;
	}
};

struct AddOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return a + b;
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return grad.Self();
	}
};

struct MulOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return a * b;
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return value.Self() * grad.Self();
	}
};

struct ReluGradOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return b >= 0.0f ? a : 0.0f;
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return value.Self() * grad.Self();
	}
};

template<typename TLhs, typename TRhs>
inline TBinaryExp<AddOp, TLhs, TRhs> operator + (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TBinaryExp<AddOp, TLhs, TRhs>(lhs.Self(), rhs.Self());
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<MulOp, TLhs, TRhs> operator * (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TBinaryExp<MulOp, TLhs, TRhs>(lhs.Self(), rhs.Self());
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<ReluGradOp, TLhs, TRhs> ReluGradExp(const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TBinaryExp<ReluGradOp, TLhs, TRhs>(lhs.Self(), rhs.Self());
}

template<typename TOp, typename TParam>
struct TUnaryExp : public TExp<TUnaryExp<TOp, TParam>>
{
	const TParam param;

	TUnaryExp(const TParam& _param)
		: param(_param.Self())
	{
	}

	TUnaryExp(const TUnaryExp& _rhs)
		: param(_rhs.param.Self())
	{
	}

	TUnaryExp(TUnaryExp&& _rhs)
		: param(Move(_rhs.param.Self()))
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		float val = TOp::Exec(param.Eval(i, broadcastIndex));
		return val;
	}

	template<typename TGrad>
	void Backward(const TExp<TGrad>& inGrad) const
	{
		const TGrad& grad = inGrad.Self();
		Assert(grad.Shape() == Shape());

		return param.Backward(TOp::Backward(param.Self(), grad));
	}

	__forceinline TensorShape Shape() const
	{
		return param.Shape();
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
		param.Preprocess(bForceRecompute);

		return;
	}
};

struct NegateOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return -val;
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Scalar(-1) * grad.Self();
	}
};

struct InvOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return 1.0f / val;
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Scalar(-1) / (value.Self() * value.Self()) * grad.Self();
	}
};

struct ExpOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Exp(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Exp(value.Self()) * grad.Self();
	}
};

struct SqrtOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Sqrt(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Scalar(0.5f) / Sqrt(value.Self()) * grad.Self();
	}
};

struct SquareOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Square(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Scalar(2.0) * value.Self() * grad.Self();
	}
};

struct LogOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Log(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Inv(value.Self()) * grad.Self();
	}
};

struct SinOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Sin(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Cos(value.Self()) * grad.Self();
	}
};

struct CosOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Cos(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return -Sin(value.Self()) * grad.Self();
	}
};

struct TanOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Tan(val);
	}

	template<typename TVal, typename TGrad>
	static auto Backward(const TExp<TVal>& value, const TExp<TGrad>& grad)
	{
		return Inv(Cos(value.Self()) * Cos(value.Self())) * grad.Self();
	}
};

struct ReluOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return val > 0.0f ? val : 0.0f;
	}
};

struct AbsOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Abs(val);
	}
};

template<typename TLhs, typename TRhs>
inline TBinaryExp<AddOp, TLhs, TUnaryExp<NegateOp, TRhs>> operator - (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TBinaryExp<AddOp, TLhs, TUnaryExp<NegateOp, TRhs>>(lhs.Self(), TUnaryExp<NegateOp, TRhs>(rhs.Self()));
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<MulOp, TLhs, TUnaryExp<InvOp, TRhs>> operator / (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TBinaryExp<MulOp, TLhs, TUnaryExp<InvOp, TRhs>>(lhs.Self(), TUnaryExp<InvOp, TRhs>(rhs.Self()));
}

template<typename TParam>
inline TUnaryExp<NegateOp, TParam> NegateExp(const TExp<TParam>& param)
{
	return TUnaryExp<NegateOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<InvOp, TParam> InvExp(const TExp<TParam>& param)
{
	return TUnaryExp<InvOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<ExpOp, TParam> ExponentExp(const TExp<TParam>& param)
{
	return TUnaryExp<ExpOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<SqrtOp, TParam> SqrtExp(const TExp<TParam>& param)
{
	return TUnaryExp<SqrtOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<SquareOp, TParam> SquareExp(const TExp<TParam>& param)
{
	return TUnaryExp<SquareOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<LogOp, TParam> LogExp(const TExp<TParam>& param)
{
	return TUnaryExp<LogOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<SinOp, TParam> SinExp(const TExp<TParam>& param)
{
	return TUnaryExp<SinOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<CosOp, TParam> CosExp(const TExp<TParam>& param)
{
	return TUnaryExp<CosOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<TanOp, TParam> TanExp(const TExp<TParam>& param)
{
	return TUnaryExp<TanOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<AbsOp, TParam> AbsExp(const TExp<TParam>& param)
{
	return TUnaryExp<AbsOp, TParam>(param.Self());
}

template<typename TParam>
inline TUnaryExp<ReluOp, TParam> ReluActivateExp(const TExp<TParam>& param)
{
	return TUnaryExp<ReluOp, TParam>(param.Self());
}

struct TConstantExp : public TExp<TConstantExp>
{
	float val;
	TensorShape shape;

	TConstantExp(const float _val, const TensorShape& _shape)
		: val(_val)
		, shape(_shape)
	{
	}

	template<typename... TShape>
	TConstantExp(const float _val, TShape... shape)
		: TConstantExp(_val, { shape... })
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		return val;
	}

	template<typename TGrad>
	void Backward(const TExp<TGrad>& inGrad) const
	{
	}

	__forceinline TensorShape Shape() const
	{
		return shape;
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
	}
};

template<typename TLhs, typename TRhs>
struct TDotExp : public TExp<TDotExp<TLhs, TRhs>>
{
	const TLhs lhs;
	const TRhs rhs;

	mutable Tensorf value;

	TDotExp(const TLhs& _lhs, const TRhs& _rhs)
		: lhs(_lhs.Self())
		, rhs(_rhs.Self())
	{
	}

	TDotExp(const TDotExp& _rhs)
		: lhs(_rhs.lhs.Self())
		, rhs(_rhs.rhs.Self())
		, value(_rhs.value)
	{
	}

	TDotExp(const TDotExp&& _rhs)
		: lhs(Move(_rhs.lhs.Self()))
		, rhs(Move(_rhs.rhs.Self()))
		, value(Move(_rhs.value))
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		return value.Eval(i, broadcastIndex);
	}

	__forceinline TensorShape Shape() const
	{
		const TensorShape& leftShape = lhs.Shape();
		const TensorShape& rightShape = rhs.Shape();

		Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
		Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

		Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

		TensorShape shape = { leftShape[0], rightShape[1] };

		return shape;
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
		lhs.Preprocess(bForceRecompute);
		rhs.Preprocess(bForceRecompute);

		if (bForceRecompute || value.Empty())
		{
			value.Resize(Shape());

			Tensorf left = lhs;
			Tensorf right = rhs;
			value = Tensorf::Dot(left, right);
		}

		return;
	}

	template<typename TGrad>
	void Backward(const TExp<TGrad>& inGrad) const
	{
		Tensorf left = lhs;
		Tensorf right = rhs;

		auto leftGrad = Dot(inGrad.Self(), right.GetTransposed());
		auto rightGrad = Dot(left.GetTransposed(), inGrad);

		lhs.Backward(leftGrad);
		rhs.Backward(rightGrad);

		return;
	}
};

template<typename TLhs, typename TRhs>
inline TDotExp<TLhs, TRhs> DotExp(const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TDotExp<TLhs, TRhs>(lhs.Self(), rhs.Self());
}

template<typename TParam>
struct TBroadcastExp : public TExp<TBroadcastExp<TParam>>
{
	const TParam param;
	const TensorShape shape;

	TBroadcastExp(const TParam& _param, const TensorShape& _shape)
		: param(_param)
		, shape(_shape)
	{
	}

	template<typename... TShape>
	TBroadcastExp(const TParam& _param, TShape... shape)
		: TBroadcastExp(_val, { shape... })
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		return param.Eval(i, broadcastIndex);
	}

	__forceinline TensorShape Shape() const
	{
		return shape;
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
		param.Preprocess(bForceRecompute);
		return;
	}
};

template<typename TOp, typename TOperand>
struct TProjectExp : public TExp<TProjectExp<TOp, TOperand>>
{
	const TOperand operand;
	TOp op;
	float initVal;
	TensorShape axes;
	bool keepDim;

	mutable Tensorf value;

	TProjectExp(const TOperand& _operand, TOp _op, const float _initVal, const TensorShape& _axes, const bool& _keepDim)
		: operand(_operand.Self())
		, op(_op)
		, initVal(_initVal)
		, axes(_axes)
		, keepDim(_keepDim)
	{
	}

	TProjectExp(const TProjectExp& exp)
		: operand(exp.operand.Self())
		, value(exp.value)
		, op(exp.op)
		, initVal(exp.initVal)
		, axes(exp.axes)
		, keepDim(exp.keepDim)
	{
	}

	TProjectExp(const TProjectExp&& exp)
		: operand(Move(exp.operand.Self()))
		, value(Move(exp.value))
		, op(exp.op)
		, initVal(exp.initVal)
		, axes(exp.axes)
		, keepDim(exp.keepDim)
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		if (axes.Size() > 0)
			return value.Eval(i, broadcastIndex);
		else
			return operand.Eval(i, broadcastIndex);
	}

	__forceinline TensorShape Shape() const
	{
		if (axes.Size() > 0)
		{
			return Tensorf::ProjectionShape(operand.Shape(), axes, keepDim);
		}
		else
			return operand.Shape();
	}

	__forceinline void Preprocess(const bool bForceRecompute = false) const
	{
		operand.Preprocess(bForceRecompute);

		if (axes.Size() > 0)
		{
			if (bForceRecompute || value.Empty())
			{
				Tensorf operandVal = operand;
				value = Tensorf::ProjectionOp<TOp>(operandVal, axes, keepDim, op, initVal);
			}
		}
	}

	template<typename TGrad>
	void Backward(const TExp<TGrad>& inGrad) const
	{
		const TGrad& grad = inGrad.Self();
		Assert(grad.Shape() == Shape());

		auto broadcast = TBroadcastExp<TGrad>(inGrad.Self(), operand.Shape());

		return operand.Backward(broadcast.Self());
	}
};


template<typename TOperand>
inline TProjectExp<Algorithm::Plus<>, TOperand> SumExp(const TExp<TOperand>& operand, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	return TProjectExp<Algorithm::Plus<>, TOperand>(operand.Self(), Algorithm::Plus<>(), 0.0f, axes, keepDim);
}

template<typename TOperand>
inline TProjectExp<Algorithm::Multiply<>, TOperand> ProductExp(const TExp<TOperand>& operand, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	return TProjectExp<Algorithm::Multiply<>, TOperand>(operand.Self(), Algorithm::Multiply<>(), 1.0f, axes, keepDim);
}

template<typename TOperand>
inline TProjectExp<Algorithm::Max<>, TOperand> MaxExp(const TExp<TOperand>& operand, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	return TProjectExp<Algorithm::Max<>, TOperand>(operand.Self(), Algorithm::Max<>(), float(Math::EDX_NEG_INFINITY), axes, keepDim);
}

template<typename TParam>
inline TUnaryExp<NegateOp, TParam> operator - (const TExp<TParam>& param)
{
	return NegateExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<InvOp, TParam> Inv(const TExp<TParam>& param)
{
	return InvExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<ExpOp, TParam> Exp(const TExp<TParam>& param)
{
	return ExponentExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<SqrtOp, TParam> Sqrt(const TExp<TParam>& param)
{
	return SqrtExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<SquareOp, TParam> Square(const TExp<TParam>& param)
{
	return SquareExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<LogOp, TParam> Log(const TExp<TParam>& param)
{
	return LogExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<SinOp, TParam> Sin(const TExp<TParam>& param)
{
	return SinExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<CosOp, TParam> Cos(const TExp<TParam>& param)
{
	return CosExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<TanOp, TParam> Tan(const TExp<TParam>& param)
{
	return TanExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<AbsOp, TParam> Abs(const TExp<TParam>& param)
{
	return AbsExp(param);
}

template<typename TParam>
__forceinline TUnaryExp<ReluOp, TParam> ReluActivate(const TExp<TParam>& param)
{
	return ReluActivateExp(param);
}

template<typename TLhs, typename TRhs>
__forceinline TBinaryExp<ReluGradOp, TLhs, TRhs> ReluGradient(const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return ReluGradExp(lhs, rhs);
}

template<typename TLhs, typename TRhs>
__forceinline TDotExp<TLhs, TRhs> Dot(const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return DotExp(lhs, rhs);
}

template<typename TOperand>
__forceinline TProjectExp<Algorithm::Plus<>, TOperand> Sum(const TExp<TOperand>& param, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	return SumExp(param, axes, keepDim);
}


template<typename TOperand>
__forceinline auto Unbroadcast(const TExp<TOperand>& tensor, const TensorShape& target)
{
	const TOperand& tens = tensor.Self();
	TensorShape shape = tens.Shape();
	//if (shape == target)
	//	return tens;

	TensorShape axes;

	if (target.LinearSize() == 1)
	{
		axes.Add(-1);
	}
	else
	{
		for (int i = 0; i < shape.Size(); i++)
		{
			if (shape[i] > target[i])
				axes.Add(i);
		}
	}

	return SumExp(tens, axes, false);
}

template<typename TOperand>
__forceinline TProjectExp<Algorithm::Multiply<>, TOperand> Product(const TExp<TOperand>& param, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	return ProductExp(param, axes, keepDim);
}

template<typename TOperand>
__forceinline TProjectExp<Algorithm::Max<>, TOperand> Max(const TExp<TOperand>& param, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	return MaxExp(param, axes, keepDim);
}

template<typename TOperand>
__forceinline auto Mean(const TExp<TOperand>& x, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	auto sum = Sum(x, axes, keepDim);

	float invDivisor = sum.Self().Shape().LinearSize() / float(x.Self().Shape().LinearSize());
	auto mean = sum * Scalar(invDivisor);

	return mean;
}

template<typename TOperand>
__forceinline auto Variance(const TExp<TOperand>& x, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	auto mean = Mean(x, axes, keepDim);
	auto centeredX = x - mean;
	auto variance = Mean(centeredX * centeredX, axes, keepDim);

	return variance;
}

template<typename TOperand>
__forceinline auto StandardDeviation(const TExp<TOperand>& x, const TensorShape& axes = { -1 }, const bool keepDim = false)
{
	auto variance = Variance(x, axes, keepDim);

	return Sqrt(variance + Scalar(1e-5f));
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