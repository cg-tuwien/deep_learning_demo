#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <memory>
#include "Eigen/Core"

using ArrayXX = Eigen::ArrayXXf;

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
    ArrayXX m_aOpb;

public:
    Expression(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b, operators::Ptr op) : m_a(a), m_b(b), m_op(op) {}
    virtual ~Expression() = default;

    virtual ArrayXX evalForward();

    virtual void differentiateBackward(ArrayXX factors = ArrayXX::Constant(1, 1, 1));

    virtual Eigen::Index rows() const;
    virtual Eigen::Index cols() const;

protected:
    Expression() {}
};

class Variable : public Expression {
    ArrayXX m_value;
    ArrayXX m_derivative;

public:
    Variable(ArrayXX v) : m_value(v), m_derivative(ArrayXX::Constant(v.rows(), v.cols(), 0)) {}

    virtual ArrayXX evalForward();
    virtual Eigen::Index rows() const { return m_value.rows(); }
    virtual Eigen::Index cols() const { return m_value.cols(); }
    virtual void differentiateBackward(ArrayXX factors);
    ArrayXX derivative() { return m_derivative; }
};

inline std::shared_ptr<Variable> make_var(ArrayXX v) { return std::make_shared<Variable>(std::move(v)); }

ExpressionPtr operator + (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr operator * (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr log(const ExpressionPtr& a);
ExpressionPtr vvt(const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr reduceSum(const ExpressionPtr &a);

#endif // EXPRESSION_H
