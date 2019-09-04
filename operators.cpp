#include "operators.h"

#include <cmath>

#include <QtGlobal>

#include "Expression.h"


namespace operators {
Add g_add;
Mul g_mul;
Log g_log;
Vvt g_vvt;
ReduceSum g_reduceSum;

ArrayXX Base::differentiateWrtA(Expression *a, Expression *b)
{
    return ArrayXX::Constant(a->rows(), a->cols(), 1);
}

ArrayXX Base::differentiateWrtB(Expression *a, Expression *b)
{
    return ArrayXX::Constant(b->rows(), b->cols(), 1);
}

ArrayXX Base::chainA(const ArrayXX &back, const ArrayXX &dA)
{
    return back * dA;
}

ArrayXX Base::chainB(const ArrayXX &back, const ArrayXX &dB)
{
    return back * dB;
}

ArrayXX Mul::differentiateWrtA(Expression*, Expression* b)
{
    return b->evalForward();
}

ArrayXX Mul::differentiateWrtB(Expression* a, Expression*)
{
    return a->evalForward();
}

ArrayXX Log::eval(const ArrayXX &a, const ArrayXX &)
{
    return Eigen::log(a);
}

ArrayXX Log::differentiateWrtA(Expression* a, Expression*)
{
    return 1.f / a->evalForward();
}

ArrayXX Vvt::eval(const ArrayXX &a, const ArrayXX &b)
{
    Q_ASSERT(a.cols() == 1);
    Q_ASSERT(b.rows() == 1);
    return a.matrix() * b.matrix();
}

ArrayXX Vvt::differentiateWrtA(Expression *a, Expression *b)
{
    return ArrayXX::Constant(a->rows(), 1, 1).matrix() * b->evalForward().matrix();
}

ArrayXX Vvt::differentiateWrtB(Expression *a, Expression *b)
{
    return a->evalForward().matrix() * ArrayXX::Constant(1, b->cols(), 1).matrix();
}

ArrayXX Vvt::chainA(const ArrayXX &back, const ArrayXX &dA)
{
    // back = a.rows x b.cols
    // dA =        - " -
    ArrayXX ret =  (back * dA).rowwise().sum();
    return ret;
}

ArrayXX Vvt::chainB(const ArrayXX &back, const ArrayXX &dB)
{
    // back = a.rows x b.cols
    // dB =        - " -
    ArrayXX ret =  (back * dB).colwise().sum();
    return ret;
}

ArrayXX ReduceSum::eval(const ArrayXX &a, const ArrayXX &b)
{
    return ArrayXX::Constant(1, 1, a.sum());
}

ArrayXX ReduceSum::chainA(const ArrayXX &back, const ArrayXX &dA)
{
    // back is 1x1
    // dA is nxm
    Q_ASSERT(back.size() == 1);
    return dA * back(0, 0);
}

ArrayXX ReduceSum::chainB(const ArrayXX &back, const ArrayXX &dB)
{
    return chainA(back, dB);
}

}


