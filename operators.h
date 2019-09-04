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
using Ptr = Base*;

struct Add : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a + b; }
};
extern Add g_add;

struct Mul : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a.array() * b.array(); }
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtB(const ArrayXX& a, const ArrayXX& b) override;
};
extern Mul g_mul;

struct Log : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
};
extern Log g_log;

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

struct ReduceSum : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB) override;
    virtual Size outSize(const Size&, const Size&) override { return {1, 1}; }
};
extern ReduceSum g_reduceSum;

struct ReduceProd : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB) override;
    virtual Size outSize(const Size&, const Size&) override { return {1, 1}; }
};
extern ReduceProd g_reduceProd;
}

#endif // OPERATORS_H
