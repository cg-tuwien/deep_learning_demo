#include "Expression.h"

#include <QtGlobal>

#include "operators.h"

ArrayXX Expression::evalForward()
{
    if (m_aOpb.size() == 0)
        m_aOpb = m_op->eval(m_a->evalForward(), m_b->evalForward());
    return m_aOpb;
}
void Expression::differentiateBackward(ArrayXX factors)
{
    m_a->differentiateBackward(m_op->chainA(factors, m_op->differentiateWrtA(m_a.get(), m_b.get())));
    m_b->differentiateBackward(m_op->chainB(factors, m_op->differentiateWrtB(m_a.get(), m_b.get())));
}

Eigen::Index Expression::rows() const
{
    Q_ASSERT(m_aOpb.size());
    return m_aOpb.rows();
}

Eigen::Index Expression::cols() const
{
    Q_ASSERT(m_aOpb.size());
    return m_aOpb.cols();
}

ArrayXX Variable::evalForward()
{
    return m_value;
}

void Variable::differentiateBackward(ArrayXX factors)
{
    m_derivative += factors;
}

ExpressionPtr operator +(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, &operators::g_add);
}

ExpressionPtr operator *(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, &operators::g_mul);
}

ExpressionPtr log(const ExpressionPtr &a)
{
    return std::make_shared<Expression>(a, make_var(ArrayXX::Constant(1, 1, 0)), &operators::g_log);
}

ExpressionPtr reduceSum(const ExpressionPtr &a)
{
    return std::make_shared<Expression>(a, make_var(ArrayXX::Constant(1, 1, 0)), &operators::g_reduceSum);
}

ExpressionPtr vvt(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, &operators::g_vvt);
}
