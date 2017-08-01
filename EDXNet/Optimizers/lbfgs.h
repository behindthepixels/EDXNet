/**
* Unconstrained Limited memory BFGS(L-BFGS).
*
* Forked from https://github.com/chokkan/liblbfgs
*
* The MIT License
*
* Copyright (c) 1990 Jorge Nocedal
* Copyright (c) 2007-2010 Naoaki Okazaki
* Copyright (c) 2014-2017 Yafei Zhang
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
#ifndef LBFGS_H_
#define LBFGS_H_

#if !defined EXTERN
#if defined __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif
#endif

/************************************************************************/
/* vector operations */
/************************************************************************/
EXTERN float* vecalloc(int size);
EXTERN void vecfree(float* vec);
EXTERN void veczero(float* x, const int n);
EXTERN void veccpy(float* y, const float* x, const int n);
EXTERN void vecncpy(float* y, const float* x, const int n);
EXTERN void vecadd(float* y, const float* x, const float c, const int n);
EXTERN void vecdiff(float* z, const float* x, const float* y, const int n);
EXTERN void vecsum(float* s, const float* x, const int n);
EXTERN void vecscale(float* y, const float c, const int n);
EXTERN void vecdot(float* s, const float* x, const float* y, const int n);
EXTERN void vec2norm(float* s, const float* x, const int n);
EXTERN void vec2norminv(float* s, const float* x, const int n);

/************************************************************************/
/* L-BFGS */
/************************************************************************/
/**
* Symbolic conventions.
*
*  x, the variables.
*  f=f(x), the object function.
*  g=g(x), the gradient of f.
*  d, the search direction in line search.
*  a, the step length in line search.
* |...|, L1 norm.
* ||...||, L2 norm.
*/

/**
* Return values of lbfgs().
*/
enum {
  /** Successful codes */
  /** L-BFGS reaches convergence. */
  LBFGS_SUCCESS = 0,
  LBFGS_CONVERGENCE = 0,
  LBFGS_CONVERGENCE_DELTA,
  /** The initial variables already minimize the objective function. */
  LBFGS_ALREADY_MINIMIZED,

  /** Error codes */
  /** Logic error. */
  LBFGSERR_LOGICERROR = -1024,
  /** The optimization process has been canceled by
  * a user-defined lbfgs_progress_t callback. */
  LBFGSERR_CANCELED,

  /** Invalid n passed to lbfgs(). */
  LBFGSERR_INVALID_N,
  /** Invalid parameter lbfgs_parameter_t::epsilon. */
  LBFGSERR_INVALID_EPSILON,
  /** Invalid parameter lbfgs_parameter_t::past. */
  LBFGSERR_INVALID_TESTPERIOD,
  /** Invalid parameter lbfgs_parameter_t::delta. */
  LBFGSERR_INVALID_DELTA,
  /** Invalid parameter lbfgs_parameter_t::linesearch. */
  LBFGSERR_INVALID_LINESEARCH,
  /** Invalid parameter lbfgs_parameter_t::max_step. */
  LBFGSERR_INVALID_MINSTEP,
  /** Invalid parameter lbfgs_parameter_t::max_step. */
  LBFGSERR_INVALID_MAXSTEP,
  /** Invalid parameter lbfgs_parameter_t::ftol. */
  LBFGSERR_INVALID_FTOL,
  /** Invalid parameter lbfgs_parameter_t::wolfe. */
  LBFGSERR_INVALID_WOLFE,
  /** Invalid parameter lbfgs_parameter_t::gtol. */
  LBFGSERR_INVALID_GTOL,
  /** Invalid parameter lbfgs_parameter_t::xtol. */
  LBFGSERR_INVALID_XTOL,
  /** Invalid parameter lbfgs_parameter_t::max_linesearch. */
  LBFGSERR_INVALID_MAXLINESEARCH,
  /** Invalid parameter lbfgs_parameter_t::orthantwise_c. */
  LBFGSERR_INVALID_ORTHANTWISE,
  /** Invalid parameter lbfgs_parameter_t::orthantwise_start. */
  LBFGSERR_INVALID_ORTHANTWISE_START,
  /** Invalid parameter lbfgs_parameter_t::orthantwise_end. */
  LBFGSERR_INVALID_ORTHANTWISE_END,

  /** The line-search step went out of the interval of uncertainty. */
  LBFGSERR_OUTOFINTERVAL,
  /** A logic error occurred; alternatively,
  * the interval of uncertainty became too small. */
  LBFGSERR_INCORRECT_TMINMAX,
  /** A rounding error occurred; alternatively, no line-search step
  * satisfies the sufficient decrease and curvature conditions. */
  LBFGSERR_ROUNDING_ERROR,
  /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
  LBFGSERR_MINIMUMSTEP,
  /** The line-search step became larger than lbfgs_parameter_t::max_step. */
  LBFGSERR_MAXIMUMSTEP,
  /** The line-search reaches the maximum number of evaluations. */
  LBFGSERR_MAXIMUMLINESEARCH,
  /** The algorithm reaches the maximum number of iterations. */
  LBFGSERR_MAXIMUMITERATION,
  /** Relative width of the interval of uncertainty is at most
  * lbfgs_parameter_t::xtol. */
  LBFGSERR_WIDTHTOOSMALL,
  /** A logic error (negative line-search step) occurred. */
  LBFGSERR_INVALIDPARAMETERS,
  /** The current search direction increases the objective function value. */
  LBFGSERR_INCREASEGRADIENT,
  /** The line search algorithm fails. */
  LBFGSERR_LINE_SEARCH_FAILED,
};

/**
* Line search algorithms.
* lbfgs_parameter_t::linesearch must be one of the following values.
*/
enum {
  /** The default algorithm (MoreThuente method). */
  LBFGS_LINESEARCH_DEFAULT = 0,
  /** MoreThuente method proposed by More and Thuente. */
  LBFGS_LINESEARCH_MORETHUENTE = 0,
  /**
  * Backtracking method with the Armijo condition.
  *  The backtracking method finds the step length such that it satisfies
  *  the sufficient decrease (Armijo) condition,
  *      - f(x + a * d) <= f(x) + lbfgs_parameter_t::ftol * a * g(x)^T d.
  */
  LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
  /**
  * Backtracking method with regular Wolfe condition.
  *  The backtracking method finds the step length such that it satisfies
  *  both the Armijo condition
  *  and the curvature condition,
  *      - g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d
  */
  LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
  /**
  * Backtracking method with strong Wolfe condition.
  *  The backtracking method finds the step length such that it satisfies
  *  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
  *  and the following condition,
  *      - |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|
  */
  LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3,
};

/**
* L-BFGS optimization parameters.
*/
typedef struct {
  /**
  * The number of corrections to approximate the inverse hessian matrix.
  *  The L-BFGS stores the computation results of previous m iterations
  *  to approximate the inverse hessian matrix of the current iteration.
  *  It controls the size of the limited memories(corrections).
  *
  *  The default value is 6.
  *  Values less than 3 are not recommended.
  *  Large values will result in excessive computing time and memory usage.
  */
  int m;

  /**
  * Epsilon for convergence test.
  *  It determines the accuracy with which the solution is to be found.
  *  lbfgs() stops when:
  *      ||g(x)|| < epsilon * max(1, ||x||)
  *
  *  The default value is 1e-5.
  */
  float epsilon;

  /**
  * Distance for delta-based convergence test.
  *  It determines the distance, in iterations, to compute the rate of decrease
  * of f(x).
  *  If the it is zero, lbfgs() does not perform the delta-based convergence
  * test.
  *
  *  The default value is 0.
  */
  int past;

  /**
  * Delta for convergence test.
  *  It determines the minimum rate of decrease of f(x).
  *  lbfgs() stops when:
  *      (f(x_{k+1}) - f(x_{k})) / f(x_{k}) < delta
  *
  *  The default value is 1e-5.
  */
  float delta;

  /**
  * The maximum number of iterations.
  *  lbfgs() stops with LBFGSERR_MAXIMUMITERATION
  *  when the iteration counter exceeds max_iterations.
  *  Zero means a never ending optimization process until convergence or errors.
  *
  *  The default value is 0.
  */
  int max_iterations;

  /**
  * The line search algorithm.
  */
  int linesearch;

  /**
  * The maximum number of trials for the line search per iteration.
  *
  *  The default value is 40.
  */
  int max_linesearch;

  /**
  * The minimum step of the line search.
  *
  *  The default value is 1e-20.
  *
  *  This value need not be modified unless
  *  the exponents are too large for the machine being used, or unless the
  *  problem is extremely badly scaled (in which case the exponents should
  *  be increased).
  */
  float min_step;

  /**
  * The maximum step of the line search.
  *
  *  The default value is 1e+20.
  *
  *  This value need not be modified unless
  *  the exponents are too large for the machine being used, or unless the
  *  problem is extremely badly scaled (in which case the exponents should
  *  be increased).
  */
  float max_step;

  /**
  * A parameter to control the accuracy of the line search.
  *
  *  The default value is 1e-4.
  *  It should be greater than zero and smaller than 0.5.
  */
  float ftol;

  /**
  * A coefficient for the (strong) Wolfe condition.
  *  It is valid only when the backtracking line-search
  *  algorithm is used with the Wolfe condition
  *  (LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
  * LBFGS_LINESEARCH_BACKTRACKING_WOLFE).
  *
  *  The default value is 0.9.
  *  It should be greater than the ftol parameter and smaller than 1.0.
  */
  float wolfe;

  /**
  * A parameter to control the accuracy of the line search.
  *
  *  The default value is 0.9.
  *  It should be greater than the ftol parameter (1e-4) and smaller than 1.0.
  *
  *  If evaluations of f(x) and g(x) are expensive
  *  with respect to the cost of the iteration (when n is very large),
  *  it may be advantageous to set it to a small value.
  *
  *  A typical small value is 0.1.
  */
  float gtol;

  /**
  * The machine precision for floating-point values.
  *  It must be a positive value set by a client program to
  *  estimate the machine precision. The line search will terminate
  *  with LBFGSERR_ROUNDING_ERROR if the relative width
  *  of the interval of uncertainty is less than it.
  */
  float xtol;

  /**
  * Coefficient for the L1 norm regularization of x.
  *  It should be set to zero for standard optimization problems.
  *  Setting it to a positive value activates OWLQN method,
  *  which minimizes f(x) + C|x|.
  *  It is the coefficient C.
  *
  *  The default value is 0.0.
  */
  float orthantwise_c;

  /**
  * Start/end index for computing |x|.
  *  They are valid only for OWLQN method (orthantwise_c != 0.0).
  *  They specify the first and last indices between which lbfgs() computes |x|.
  *
  *  The default values are 0/-1.
  */
  int orthantwise_start;
  int orthantwise_end;
} lbfgs_parameter_t;

/**
* Callback to provide f(x) and g(x).
* Obviously, a client program MUST implement it.
*
*  @param  instance     User data passed to lbfgs().
*  @param  n            The dimension of x.
*  @param  x            Current x.
*  @param  g            [OUTPUT] Current g(x).
*  @param  step         The current step of the line search.
*  @retval float       f(x).
*/
typedef float (*lbfgs_evaluate_t)(void* instance, const int n, const float* x,
                                   float* g, const float step);

/**
* Callback to receive the optimization progress.
*
*  @param  instance     User data passed to lbfgs().
*  @param  n            The dimension of x.
*  @param  x            Current x.
*  @param  g            Current g(x).
*  @param  fx           Current f(x).
*  @param  xnorm        The Euclidean norm of the x.
*  @param  gnorm        The Euclidean norm of the g.
*  @param  step         The line-search step used for this iteration.
*  @param  k            The iteration count.
*  @param  n_evaluate   The number of evaluations of f(x) and g(x).
*  @retval int          Zero to continue the optimization process.
*                       Non-zero value will cancel the optimization process.
*                       Default progress callback never always returns zero.
*/
typedef int (*lbfgs_progress_t)(void* instance, int n, const float* x,
                                const float* g, const float fx,
                                const float xnorm, const float gnorm,
                                const float step, int k, int n_evaluate);

/**
* Start a L-BFGS optimization.
*
*  @param  n            The dimension of x.
*  @param  x            The initial x.
*                       It must be allocated by vecalloc().
*  @param  pfx          The pointer to the variable that receives the final
* f(x).
*                       It can be set to NULL if the final f(x) is unnecessary.
*  @param  evaluate     The callback function to provide f(x) and g(x).
*  @param  progress     The callback function to receive the progress.
*                       This argument can be set to NULL, then a default one
* will be used.
*  @param  instance     User data.
*                       'evaluate' and 'progress' will receive it.
*  @param  param        A pointer to lbfgs_parameter_t structure.
*                       It can be set to NULL to use the default parameters.
*                       It is recommended to obtain it by
* lbfgs_default_parameter(),
*                       and then some fields can be overwrite manually.
*  @retval int          The status code.
*                       A positive value or zero indicates a success.
*                       A negative value indicates an error.
*/
EXTERN int lbfgs(int n, float* x, float* pfx, lbfgs_evaluate_t evaluate,
                 lbfgs_progress_t progress, void* instance,
                 const lbfgs_parameter_t* param);

/**
* Initialize L-BFGS parameters to the default values.
*/
EXTERN void lbfgs_default_parameter(lbfgs_parameter_t* param);

#endif /* LBFGS_H_ */
