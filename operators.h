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
using Ptr = std::shared_ptr<Base>;

struct Add : public Base {
    virtual float eval(const float& a, const float& b) { return a + b; }
    virtual float differentiateWrtA(const Expression& a, const Expression& b);
    virtual float differentiateWrtB(const Expression& a, const Expression& b);
};

struct Mul : public Base {
    virtual float eval(const float& a, const float& b) { return a * b; }
    virtual float differentiateWrtA(const Expression& a, const Expression& b);
    virtual float differentiateWrtB(const Expression& a, const Expression& b);
};

struct Log : public Base {
    virtual float eval(const float& a, const float& b);
    virtual float differentiateWrtA(const Expression& a, const Expression& b);
    virtual float differentiateWrtB(const Expression& a, const Expression& b);
};

}

#endif // OPERATORS_H
