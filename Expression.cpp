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

#include "Expression.h"

#include <iostream>

#include <QtGlobal>

#include "operators.h"

Expression::Expression(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b, operators::Ptr op) : m_a(a), m_b(b), m_op(op)
{
}

ArrayXX Expression::evalForward()
{
    if (!m_aOpbValid) {
        m_aOpb = m_op->eval(m_a->evalForward(), m_b->evalForward());
        m_aOpbValid = true;
//		if (m_aOpb.isInf().any() || m_aOpb.isNaN().any()) {
//			std::cout << "a=" << m_a->evalForward().transpose() << std::endl;
//			std::cout << "b=" << m_b->evalForward().transpose() << std::endl;
//		}
        Q_ASSERT(!m_aOpb.isNaN().any());
		Q_ASSERT(!m_aOpb.isInf().any());
    }
    return m_aOpb;
}
void Expression::differentiateBackward(const ArrayXX& factors)
{
//    static bool debug1 = false;
//    static bool debug2 = false;
    auto diffWrtA = m_op->differentiateWrtA(m_a->evalForward(), m_b->evalForward());
    Q_ASSERT(!diffWrtA.isNaN().any());
    Q_ASSERT(!diffWrtA.isInf().any());
    auto chainedA = m_op->chainA(factors, diffWrtA);
    Q_ASSERT(!chainedA.isNaN().any());
    Q_ASSERT(!chainedA.isInf().any());
//    if (debug1) {
//        std::cout << "factors = " << factors.transpose() << std::endl;
//        std::cout << "diffWrtA = " << diffWrtA.transpose() << std::endl;
//        std::cout << "chainedA = " << chainedA.transpose() << std::endl;
//    }
    m_a->differentiateBackward(chainedA);
//    if (debug2) {
//        std::cout << "factors = " << factors.transpose() << std::endl;
//        std::cout << "diffWrtA = " << diffWrtA.transpose() << std::endl;
//        std::cout << "chainedA = " << chainedA.transpose() << std::endl;
//    }

    auto diffWrtB = m_op->differentiateWrtB(m_a->evalForward(), m_b->evalForward());
    Q_ASSERT(!diffWrtB.isNaN().any());
    Q_ASSERT(!diffWrtB.isInf().any());
    auto chainedB = m_op->chainB(factors, diffWrtB);
    Q_ASSERT(!chainedB.isNaN().any());
    Q_ASSERT(!chainedB.isInf().any());
//    if (debug1) {
//        std::cout << "factors = " << factors.transpose() << std::endl;
//        std::cout << "diffWrtB = " << diffWrtB.transpose() << std::endl;
//        std::cout << "chainedB = " << chainedB.transpose() << std::endl;
//    }
    m_b->differentiateBackward(chainedB);
//    if (debug2) {
//        std::cout << "factors = " << factors.transpose() << std::endl;
//        std::cout << "diffWrtB = " << diffWrtB.transpose() << std::endl;
//        std::cout << "chainedB = " << chainedB.transpose() << std::endl;
//    }
}

Size Expression::size()
{
    if (m_size(0) == -1)
        m_size = m_op->outSize(m_a->size(), m_b->size());
    return m_size;
}

void Expression::reset()
{
    m_aOpbValid = false;
    if (m_a)
        m_a->reset();
    if (m_b)
        m_b->reset();
}

ArrayXX Variable::evalForward()
{
    return m_value;
}

void Variable::resetGradient()
{
    m_gradient = ArrayXX::Constant(m_value.rows(), m_value.cols(), 0);
}

void Variable::differentiateBackward(const ArrayXX& factors)
{
    m_gradient += factors;
}

ExpressionPtr operator +(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, &operators::g_add);
}

ExpressionPtr operator -(const ExpressionPtr& a, const ExpressionPtr& b)
{
    return std::make_shared<Expression>(a, b, &operators::g_subtract);
}

ExpressionPtr operator *(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, &operators::g_matMul);
}

ExpressionPtr log(const ExpressionPtr &a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_log);
}

ExpressionPtr exp(const ExpressionPtr& a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_exp);
}

ExpressionPtr normExp(const ExpressionPtr& a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_normExp);
}

ExpressionPtr relu(const ExpressionPtr& a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_relu);
}

ExpressionPtr vvt(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, &operators::g_vvt);
}

ExpressionPtr reduceSum(const ExpressionPtr &a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_reduceSum);
}

ExpressionPtr reduceProd(const ExpressionPtr &a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_reduceProd);
}

ExpressionPtr matmul(const ExpressionPtr &a, const ExpressionPtr &b)
{
    Q_ASSERT(a->rows() > 0);
    Q_ASSERT(a->cols() > 0);
    Q_ASSERT(b->rows() == a->cols());
    Q_ASSERT(b->cols() > 0);
    return std::make_shared<Expression>(a, b, &operators::g_matMul);
}

ExpressionPtr cwisemul(const ExpressionPtr& a, const ExpressionPtr& b)
{
    return std::make_shared<Expression>(a, b, &operators::g_mul);
}

ExpressionPtr cwisediv(const ExpressionPtr& a, const ExpressionPtr& b)
{
    return std::make_shared<Expression>(a, b, &operators::g_div);
}
