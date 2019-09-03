#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <memory>

class Expression;
using ExpressionPtr = std::shared_ptr<Expression>;

namespace operators {
struct Base;
using Ptr = Base*;
}

class Expression {
    ExpressionPtr m_a;
    ExpressionPtr m_b;
    operators::Ptr m_op;
    float m_aOpb;

public:
    Expression(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b, operators::Ptr op) : m_a(a), m_b(b), m_op(op) {}
    virtual ~Expression() = default;

    virtual float evalForward();
    virtual float evalForward() const;

    virtual void differentiateBackward(float factors = 1);

protected:
    Expression() {}
};

class Variable : public Expression {
    float m_value;
    float m_derivative;

public:
    Variable(float v) : m_value(v), m_derivative(0) {}

    virtual float evalForward();
    virtual float evalForward() const;
    virtual void differentiateBackward(float factors);
    float derivative() { return m_derivative; }
};

inline std::shared_ptr<Variable> make_var(float v) { return std::make_shared<Variable>(v); }

ExpressionPtr operator + (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr operator * (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr log(const ExpressionPtr& a);

#endif // EXPRESSION_H
