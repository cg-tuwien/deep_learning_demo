/*
 * Copyright (c) 2019, Adam Celarek | Research Unit of Computer Graphics | TU Wien
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of mosquitto nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "operators.h"

#include <cmath>
#include <iostream>

#include <QtGlobal>

namespace operators {
Add g_add;
Subtract g_subtract;
Mul g_mul;
Div g_div;
Log g_log;
Exp g_exp;
NormExp g_normExp;
Relu g_relu;
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

ArrayXX UnaryBase::chainB(const ArrayXX&, const ArrayXX& dB)
{
    return dB;
}

Size UnaryBase::outSize(const Size& sizeA, const Size&)
{
    return sizeA;
}

ArrayXX Subtract::differentiateWrtB(const ArrayXX&, const ArrayXX& b)
{
    return ArrayXX::Constant(b.rows(), b.cols(), -1);
}

ArrayXX Mul::differentiateWrtA(const ArrayXX&, const ArrayXX& b)
{
    return b;
}

ArrayXX Mul::differentiateWrtB(const ArrayXX& a, const ArrayXX&)
{
    return a;
}

ArrayXX Div::differentiateWrtA(const ArrayXX&, const ArrayXX& b)
{
    return 1 / b;
}

ArrayXX Div::differentiateWrtB(const ArrayXX& a, const ArrayXX& b)
{
    return -a / (b * b);
}

ArrayXX Log::eval(const ArrayXX& a, const ArrayXX&)
{
    return Eigen::log(a);
}

ArrayXX Log::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
    return 1.f / a;
}

ArrayXX Exp::eval(const ArrayXX& a, const ArrayXX&)
{
    return a.exp();
}

ArrayXX Exp::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
    return a.exp();
}

ArrayXX NormExp::eval(const ArrayXX& a, const ArrayXX& )
{
//	std::cout << "NormExp: " << a.transpose() - a.maxCoeff() << std::endl;
//	std::cout << "NormExp: " << (a.transpose() - a.maxCoeff()).exp() << std::endl;
	return (a - a.maxCoeff()).exp();
}

ArrayXX NormExp::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
	return (a - a.maxCoeff()).exp();
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

ArrayXX Relu::eval(const ArrayXX& a, const ArrayXX&)
{
//	std::cout << "relu input: " << a.transpose() << std::endl;
    return a.max(a * 0.01f);
}

ArrayXX Relu::differentiateWrtA(const ArrayXX& a, const ArrayXX&)
{
    return (a > 0.f).cast<float>() * 0.99 + 0.01;
}

}


