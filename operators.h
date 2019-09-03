#ifndef OPERATORS_H
#define OPERATORS_H

#include <memory>

class Expression;

namespace operators {
struct Base {
    virtual ~Base() = default;
    virtual float eval(const float& a, const float& b) = 0;
    virtual float differentiateWrtA(const Expression& a, const Expression& b) = 0;
    virtual float differentiateWrtB(const Expression& a, const Expression& b) = 0;
};
using Ptr = Base*;

struct Add : public Base {
    virtual float eval(const float& a, const float& b) { return a + b; }
    virtual float differentiateWrtA(const Expression& a, const Expression& b);
    virtual float differentiateWrtB(const Expression& a, const Expression& b);
};
extern Add g_add;

struct Mul : public Base {
    virtual float eval(const float& a, const float& b) { return a * b; }
    virtual float differentiateWrtA(const Expression& a, const Expression& b);
    virtual float differentiateWrtB(const Expression& a, const Expression& b);
};
extern Mul g_mul;

struct Log : public Base {
    virtual float eval(const float& a, const float& b);
    virtual float differentiateWrtA(const Expression& a, const Expression& b);
    virtual float differentiateWrtB(const Expression& a, const Expression& b);
};
extern Log g_log;

}

#endif // OPERATORS_H
