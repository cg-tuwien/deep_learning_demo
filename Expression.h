#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <memory>
#include "Eigen/Core"

using ArrayXX = Eigen::ArrayXXf;
using Size = Eigen::Vector2i;

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
    bool m_aOpbValid = false;
    Size m_size = Size(-1, -1);
public:
    Expression(std::shared_ptr<Expression> a, std::shared_ptr<Expression> b, operators::Ptr op);
    virtual ~Expression() = default;
    virtual ArrayXX evalForward();
    virtual void differentiateBackward(const ArrayXX& factors = ArrayXX::Constant(1, 1, 1));
    virtual Size size();
    Eigen::Index rows() { return this->size()(0); }
    Eigen::Index cols() { return this->size()(1); }
    void reset();
protected:
    Expression() {}
};

class Variable : public Expression {
    ArrayXX m_value;
    ArrayXX m_gradient;

public:
    Variable(ArrayXX v) : m_value(v), m_gradient(ArrayXX::Constant(v.rows(), v.cols(), 0)) {}
    Variable(Eigen::Index rows, Eigen::Index cols) : m_value(ArrayXX(rows, cols)), m_gradient(ArrayXX::Constant(rows, cols, 0)) {}

    virtual ArrayXX evalForward() override;
    virtual Size size() override { return {m_value.rows(), m_value.cols()}; }
    virtual void differentiateBackward(const ArrayXX& factors) override;
    ArrayXX& value() { return m_value; }
    void resetGradient();
    ArrayXX gradient() { return m_gradient; }

    static inline std::shared_ptr<Variable> make(ArrayXX v) { return std::make_shared<Variable>(std::move(v)); }
    static inline std::shared_ptr<Variable> make(Eigen::Index rows, Eigen::Index cols) { return std::make_shared<Variable>(rows, cols); }
    static inline std::shared_ptr<Variable> make(Eigen::Index rows, Eigen::Index cols, float value) { return make(ArrayXX::Constant(rows, cols, value)); }
    static inline std::shared_ptr<Variable> make(float value) { return make(1, 1, value); }
};
using VariablePtr = std::shared_ptr<Variable>;

class Constant : public Variable {
    virtual void differentiateBackward(const ArrayXX&) override {}
};

using ConstantPtr = VariablePtr;


ExpressionPtr operator + (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr operator - (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr operator * (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr log(const ExpressionPtr& a);
ExpressionPtr exp(const ExpressionPtr& a);
ExpressionPtr normExp(const ExpressionPtr& a);
ExpressionPtr relu(const ExpressionPtr& a);
ExpressionPtr softmax(const ExpressionPtr& a);
ExpressionPtr vvt(const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr cwisemul (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr cwisediv (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr matmul(const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr reduceSum(const ExpressionPtr &a);
ExpressionPtr reduceProd(const ExpressionPtr &a);

#endif // EXPRESSION_H
