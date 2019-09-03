#include "operators.h"
#include "Expression.h"

#include <cmath>

float operators::Add::differentiateWrtA(const Expression&, const Expression&)
{
    return 1;
}

float operators::Add::differentiateWrtB(const Expression&, const Expression&)
{
    return 1;
}

float operators::Mul::differentiateWrtA(const Expression&, const Expression& b)
{
    return b.evalForward();
}

float operators::Mul::differentiateWrtB(const Expression& a, const Expression&)
{
    return a.evalForward();
}

float operators::Log::eval(const float &a, const float &b)
{
    return std::log(a);
}

float operators::Log::differentiateWrtA(const Expression &a, const Expression &b)
{
    return 1.f / a.evalForward();
}

float operators::Log::differentiateWrtB(const Expression &a, const Expression &b)
{
    return 1.f;
}
