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
        Q_ASSERT(!m_aOpb.isNaN().any());
        Q_ASSERT(!m_aOpb.isInf().any());
    }
    return m_aOpb;
}
void Expression::differentiateBackward(const ArrayXX& factors)
{
    auto diffWrtA = m_op->differentiateWrtA(m_a->evalForward(), m_b->evalForward());
    Q_ASSERT(!diffWrtA.isNaN().any());
    Q_ASSERT(!diffWrtA.isInf().any());
    auto chainedA = m_op->chainA(factors, diffWrtA);
    Q_ASSERT(!chainedA.isNaN().any());
    Q_ASSERT(!chainedA.isInf().any());
    m_a->differentiateBackward(chainedA);


    auto diffWrtB = m_op->differentiateWrtB(m_a->evalForward(), m_b->evalForward());
    Q_ASSERT(!diffWrtB.isNaN().any());
    Q_ASSERT(!diffWrtB.isInf().any());
    auto chainedB = m_op->chainB(factors, diffWrtB);
    Q_ASSERT(!chainedB.isNaN().any());
    Q_ASSERT(!chainedB.isInf().any());
    m_b->differentiateBackward(chainedB);
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

ExpressionPtr relu(const ExpressionPtr& a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_relu);
}

ExpressionPtr softmax(const ExpressionPtr& a)
{
    return std::make_shared<Expression>(a, Constant::make(0), &operators::g_softmax);
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
