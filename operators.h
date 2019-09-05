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

struct Softmax : public UnaryBase {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
};
extern Softmax g_softmax;
}

#endif // OPERATORS_H
