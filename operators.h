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

#ifndef OPERATORS_H
#define OPERATORS_H

#include <memory>
#include "Eigen/Core"

using ArrayXX = Eigen::ArrayXXf;
using Size = Eigen::Vector2i;

namespace operators {
struct Base {
    virtual ~Base() = default;
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) = 0;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b);
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b);
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA);
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB);
    virtual Size outSize(const Size& sizeA, const Size& sizeB);
};
struct UnaryBase : public Base {
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB);
    virtual Size outSize(const Size& sizeA, const Size& sizeB);
};
using Ptr = Base*;

struct Add : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a + b; }
};
extern Add g_add;

struct Subtract : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a - b; }
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b) override;
};
extern Subtract g_subtract;

struct Mul : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a * b; }
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b) override;
};
extern Mul g_mul;

struct Div : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a / b; }
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b) override;
};
extern Div g_div;

struct Log : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
};
extern Log g_log;

struct Exp : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
};
extern Exp g_exp;

struct NormExp : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
};
extern NormExp g_normExp;

struct Vvt : public Base { // vector vector.transpose
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB) override;
};
extern Vvt g_vvt;


struct MatMul : public Base { // vector vector.transpose
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB) override;
};
extern MatMul g_matMul;

struct ReduceSum : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual Size outSize(const Size&, const Size&) override { return {1, 1}; }
};
extern ReduceSum g_reduceSum;

struct ReduceProd : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual Size outSize(const Size&, const Size&) override { return {1, 1}; }
};
extern ReduceProd g_reduceProd;

struct Relu : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
};
extern Relu g_relu;
}

#endif // OPERATORS_H
