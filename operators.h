#ifndef OPERATORS_H
#define OPERATORS_H

#include <memory>
#include "Eigen/Core"

class Expression;
using ArrayXX = Eigen::ArrayXXf;

namespace operators {
struct Base {
    virtual ~Base() = default;
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) = 0;
    virtual ArrayXX differentiateWrtA(Expression* a, Expression* b);
    virtual ArrayXX differentiateWrtB(Expression* a, Expression* b);
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA);
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB);
};
using Ptr = Base*;

struct Add : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a + b; }
};
extern Add g_add;

struct Mul : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override { return a.array() * b.array(); }
    virtual ArrayXX differentiateWrtA(Expression* a, Expression* b) override;
    virtual ArrayXX differentiateWrtB(Expression* a, Expression* b) override;
};
extern Mul g_mul;

struct Log : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(Expression* a, Expression* b) override;
};
extern Log g_log;

struct Vvt : public Base { // vector vector.transpose
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX differentiateWrtA(Expression* a, Expression* b) override;
    virtual ArrayXX differentiateWrtB(Expression* a, Expression* b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB) override;
};
extern Vvt g_vvt;

struct ReduceSum : public Base {
    virtual ArrayXX eval(const ArrayXX& a, const ArrayXX& b) override;
    virtual ArrayXX chainA(const ArrayXX& back, const ArrayXX& dA) override;
    virtual ArrayXX chainB(const ArrayXX& back, const ArrayXX& dB) override;
};
extern ReduceSum g_reduceSum;
}

#endif // OPERATORS_H
