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
    }
    return m_aOpb;
}
void Expression::differentiateBackward(const ArrayXX& factors)
{
    auto diffWrtA = m_op->differentiateWrtA(m_a->evalForward(), m_b->evalForward());
    auto chainedA = m_op->chainA(factors, diffWrtA);
    m_a->differentiateBackward(chainedA);

    auto diffWrtB = m_op->differentiateWrtB(m_a->evalForward(), m_b->evalForward());
    auto chainedB = m_op->chainB(factors, diffWrtB);
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

void Variable::resetDerivative()
{
    m_derivative = ArrayXX::Constant(m_value.rows(), m_value.cols(), 0);
}

void Variable::differentiateBackward(const ArrayXX& factors)
{
    m_derivative += factors;
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
