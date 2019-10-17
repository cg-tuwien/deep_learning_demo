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
public:
	Constant(ArrayXX v) : Variable(v) {}
	Constant(Eigen::Index rows, Eigen::Index cols) : Variable(rows, cols) {}
    virtual void differentiateBackward(const ArrayXX&) override {}
	static inline std::shared_ptr<Variable> make(ArrayXX v) { return std::make_shared<Constant>(std::move(v)); }
	static inline std::shared_ptr<Variable> make(Eigen::Index rows, Eigen::Index cols) { return std::make_shared<Constant>(rows, cols); }
	static inline std::shared_ptr<Variable> make(Eigen::Index rows, Eigen::Index cols, float value) { return make(ArrayXX::Constant(rows, cols, value)); }
	static inline std::shared_ptr<Variable> make(float value) { return make(1, 1, value); }
};

using ConstantPtr = VariablePtr;


ExpressionPtr operator + (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr operator - (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr operator * (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr log(const ExpressionPtr& a);
ExpressionPtr exp(const ExpressionPtr& a);
ExpressionPtr normExp(const ExpressionPtr& a);
ExpressionPtr relu(const ExpressionPtr& a);
ExpressionPtr vvt(const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr cwisemul (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr cwisediv (const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr matmul(const ExpressionPtr& a, const ExpressionPtr& b);
ExpressionPtr reduceSum(const ExpressionPtr &a);
ExpressionPtr reduceProd(const ExpressionPtr &a);

#endif // EXPRESSION_H
