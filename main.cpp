#include <iostream>
#include "Expression.h"

int main(int argc, char *argv[])
{
    auto x = make_var(2);
    auto y = make_var(3);

    auto expr = log(x*x + x*y + y*y)*x*y;
//    auto expr = x * y;
    auto forwardResult = expr->evalForward();

    std::cout << forwardResult << std::endl;
    expr->differentiateBackward();
    std::cout << x->derivative() << std::endl;
    std::cout << y->derivative() << std::endl;

    return 0;
}
