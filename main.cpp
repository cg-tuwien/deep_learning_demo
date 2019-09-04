#include <iostream>
#include "Eigen/Core"
#include "Expression.h"

int main(int argc, char *argv[])
{
    auto x = Variable::make(2, 1);
    x->value() << 1, 2;
    auto y = Variable::make(1, 2);
    y->value() << 3, 4;

    auto W = Variable::make(2, 2);
    W->value() << 1.1f, 1.2f,
                  1.3f, 1.4f;

//    auto x = make_var(Mat::Random(2, 2));
//    auto y = make_var(Mat::Random(2, 2));

//    auto expr = reduceSum(log(x*x + x*y + y*y)*x*y);
//    auto expr = x * y;
//    auto expr = reduceProd(matmul(x, y) * matmul(x, y));
    auto expr = reduceProd(W * matmul(matmul(W, x), y));
//    auto expr = reduceSum(matmul(W, x));
    auto forwardResult = expr->evalForward();

    std::cout << "x = \n" << x->evalForward() << std::endl;
    std::cout << "y = \n" << y->evalForward() << std::endl;
    std::cout << "W = \n" << W->evalForward() << std::endl;

    std::cout << "f(x, y) = \n" << forwardResult << std::endl;
    expr->differentiateBackward();
    std::cout << "df(x,y)/dx = \n" << x->derivative() << std::endl;
    std::cout << "df(x,y)/dy = \n" << y->derivative() << std::endl;

    return 0;
}
