#include "Expression.h"

#include <QtGlobal>

#include "operators.h"

float Expression::evalForward()
{
    m_aOpb = m_op->eval(m_a->evalForward(), m_b->evalForward());
    return m_aOpb;
}

float Expression::evalForward() const
{
    return m_aOpb;
}

void Expression::differentiateBackward(float factors)
{
    m_a->differentiateBackward(factors * m_op->differentiateWrtA(*m_a, *m_b));
    m_b->differentiateBackward(factors * m_op->differentiateWrtB(*m_a, *m_b));
}

float Variable::evalForward()
{
    return m_value;
}

float Variable::evalForward() const
{
    return m_value;
}

void Variable::differentiateBackward(float factors)
{
    m_derivative += factors;
}

ExpressionPtr operator +(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, std::make_shared<operators::Add>());
}

ExpressionPtr operator *(const ExpressionPtr &a, const ExpressionPtr &b)
{
    return std::make_shared<Expression>(a, b, std::make_shared<operators::Mul>());
}

ExpressionPtr log(const ExpressionPtr &a)
{
    return std::make_shared<Expression>(a, make_var(0), std::make_shared<operators::Log>());
}
