#pragma once

// Template expression definition
// This file should be included before Tensor class declaration in Tensor.h

template<typename ExpType>
struct TExp
{
	__forceinline const ExpType& Self() const
	{
		return *static_cast<const ExpType*>(this);
	}
};

template<typename T>
struct TScalarExp : public TExp<TScalarExp<T>>
{
	const T val;

	TScalarExp(const T& _val)
		: val(_val)
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		return val;
	}

	__forceinline TensorShape Shape() const
	{
		return{ 1 };
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

	TBinaryExp(TBinaryExp&& _rhs)
		: lhs(Move(_rhs.lhs.Self()))
		, rhs(Move(_rhs.rhs.Self()))
	{
	}

	// evaluation function, evaluate this expression at position i
	TENSOR_INLINE float Eval(const int i, const TensorParams& broadcastIndex) const
	{
		return TOp::Exec(lhs.Eval(i, broadcastIndex), rhs.Eval(i, broadcastIndex));
	}

	__forceinline TensorShape Shape() const
	{
		return BroadcastShape(lhs.Shape(), rhs.Shape());
	}
};

struct AddOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return a + b;
	}
};

struct MinusOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return a - b;
	}
};

struct MulOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return a * b;
	}
};

struct DivOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return a / b;
	}
};

struct ReluGradOp
{
	TENSOR_INLINE static float Exec(float a, float b)
	{
		return b >= 0.0f ? a : 0.0f;
	}
};

// template binary operation, works for any expressions
template<typename TOp, typename TLhs, typename TRhs>
inline TBinaryExp<TOp, TLhs, TRhs> ElementWiseBinaryOpExpression(const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return TBinaryExp<TOp, TLhs, TRhs>(lhs.Self(), rhs.Self());
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<AddOp, TLhs, TRhs> operator + (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return ElementWiseBinaryOpExpression<AddOp>(lhs, rhs);
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<MinusOp, TLhs, TRhs> operator - (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return ElementWiseBinaryOpExpression<MinusOp>(lhs, rhs);
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<MulOp, TLhs, TRhs> operator * (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return ElementWiseBinaryOpExpression<MulOp>(lhs, rhs);
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<DivOp, TLhs, TRhs> operator / (const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return ElementWiseBinaryOpExpression<DivOp>(lhs, rhs);
}

template<typename TLhs, typename TRhs>
inline TBinaryExp<ReluGradOp, TLhs, TRhs> ReluGradExp(const TExp<TLhs>& lhs, const TExp<TRhs>& rhs)
{
	return ElementWiseBinaryOpExpression<ReluGradOp>(lhs, rhs);
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
		return TOp::Exec(param.Eval(i, broadcastIndex));
	}

	__forceinline TensorShape Shape() const
	{
		return param.Shape();
	}
};

// template binary operation, works for any expressions
template<typename TOp, typename TParam>
inline TUnaryExp<TOp, TParam> ElementWiseUnaryOpExpression(const TExp<TParam>& param)
{
	return TUnaryExp<TOp, TParam>(param.Self());
}

struct ExpOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Exp(val);
	}
};

struct SqrtOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Sqrt(val);
	}
};

struct SquareOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Square(val);
	}
};

struct LogOp
{
	TENSOR_INLINE static float Exec(float val)
	{
		return Math::Log(val);
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


template<typename TParam>
inline TUnaryExp<ExpOp, TParam> ExponentExp(const TExp<TParam>& param)
{
	return ElementWiseUnaryOpExpression<ExpOp>(param);
}

template<typename TParam>
inline TUnaryExp<SqrtOp, TParam> SqrtExp(const TExp<TParam>& param)
{
	return ElementWiseUnaryOpExpression<SqrtOp>(param);
}

template<typename TParam>
inline TUnaryExp<SquareOp, TParam> SquareExp(const TExp<TParam>& param)
{
	return ElementWiseUnaryOpExpression<SquareOp>(param);
}

template<typename TParam>
inline TUnaryExp<LogOp, TParam> LogExp(const TExp<TParam>& param)
{
	return ElementWiseUnaryOpExpression<LogOp>(param);
}

template<typename TParam>
inline TUnaryExp<AbsOp, TParam> AbsExp(const TExp<TParam>& param)
{
	return ElementWiseUnaryOpExpression<AbsOp>(param);
}

template<typename TParam>
inline TUnaryExp<ReluOp, TParam> ReluActivateExp(const TExp<TParam>& param)
{
	return ElementWiseUnaryOpExpression<ReluOp>(param);
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

	__forceinline TensorShape Shape() const
	{
		return shape;
	}
};