#include "operators.h"

#include <cmath>

#include <QtGlobal>

namespace operators {
Add g_add;
Mul g_mul;
Log g_log;
Vvt g_vvt;
MatMul g_matMul;
ReduceSum g_reduceSum;
ReduceProd g_reduceProd;

ArrayXX Base::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
    return ArrayXX::Constant(a.rows(), a.cols(), 1);
}

ArrayXX Base::differentiateWrtB(const ArrayXX&, const ArrayXX& b)
{
    return ArrayXX::Constant(b.rows(), b.cols(), 1);
}

ArrayXX Base::chainA(const ArrayXX& back, const ArrayXX& dA)
{
    return back * dA;
}

ArrayXX Base::chainB(const ArrayXX& back, const ArrayXX& dB)
{
    return back * dB;
}

Size Base::outSize(const Size& sizeA, const Size& sizeB)
{
    // works for element wise and matmul/vvt
    return Size(sizeA(0), sizeB(1));
}

ArrayXX Mul::differentiateWrtA(const ArrayXX&, const ArrayXX& b)
{
    return b;
}

ArrayXX Mul::differentiateWrtB(const ArrayXX& a, const ArrayXX&)
{
    return a;
}

ArrayXX Log::eval(const ArrayXX& a, const ArrayXX&)
{
    return Eigen::log(a);
}

ArrayXX Log::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
    return 1.f / a;
}

ArrayXX Vvt::eval(const ArrayXX& a, const ArrayXX& b)
{
    Q_ASSERT(a.cols() == 1);
    Q_ASSERT(b.rows() == 1);
    return a.matrix() * b.matrix();
}

ArrayXX Vvt::differentiateWrtA(const ArrayXX& a, const ArrayXX& b)
{
    return ArrayXX::Constant(a.rows(), 1, 1).matrix() * b.matrix();
}

ArrayXX Vvt::differentiateWrtB(const ArrayXX& a, const ArrayXX& b)
{
    return a.matrix() * ArrayXX::Constant(1, b.cols(), 1).matrix();
}

ArrayXX Vvt::chainA(const ArrayXX& back, const ArrayXX& dA)
{
    // back = a.rows x b.cols
    // dA =        - " -
    // ret is a.rows x 1
    ArrayXX ret =  (back * dA).rowwise().sum();
    return ret;
}

ArrayXX Vvt::chainB(const ArrayXX& back, const ArrayXX& dB)
{
    // back = a.rows x b.cols
    // dB =        - " -
    // ret is 1 x b.cols
    ArrayXX ret =  (back * dB).colwise().sum();
    return ret;
}

ArrayXX ReduceSum::eval(const ArrayXX& a, const ArrayXX&)
{
    return ArrayXX::Constant(1, 1, a.sum());
}

ArrayXX ReduceSum::chainA(const ArrayXX& back, const ArrayXX& dA)
{
    // back is 1 x 1
    // dA is n x m
    // ret is n x m
    Q_ASSERT(back.size() == 1);
    return dA * back(0, 0);
}

ArrayXX ReduceSum::chainB(const ArrayXX& back, const ArrayXX& dB)
{
    return chainA(back, dB);
}

ArrayXX ReduceProd::eval(const ArrayXX& a, const ArrayXX&)
{
    return ArrayXX::Constant(1, 1, a.prod());
}

ArrayXX ReduceProd::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
    return ArrayXX::Constant(a.rows(), a.cols(), a.prod()) / a;
}

ArrayXX ReduceProd::chainA(const ArrayXX& back, const ArrayXX& dA)
{
    // back is 1 x 1
    // dA is n x m
    // ret is n x m
    Q_ASSERT(back.size() == 1);
    return dA * back(0, 0);
}

ArrayXX ReduceProd::chainB(const ArrayXX& back, const ArrayXX& dB)
{
    return chainA(back, dB);
}

ArrayXX MatMul::eval(const ArrayXX& a, const ArrayXX& b)
{
    return a.matrix() * b.matrix();
}

ArrayXX MatMul::differentiateWrtA(const ArrayXX&, const ArrayXX& b)
{
    return b;
}

ArrayXX MatMul::differentiateWrtB(const ArrayXX& a, const ArrayXX&)
{
    return a;
}

ArrayXX MatMul::chainA(const ArrayXX& back, const ArrayXX& dA)
{
    return back.matrix() * dA.matrix().transpose();
}

ArrayXX MatMul::chainB(const ArrayXX& back, const ArrayXX& dB)
{
    return dB.matrix().transpose() * back.matrix();
}

}


