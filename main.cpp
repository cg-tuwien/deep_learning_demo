#include <iostream>
#include "Eigen/Core"
#include "Expression.h"

int main(int argc, char *argv[])
{
    auto xRaw = ArrayXX(2, 1);
    xRaw << 1, 2;
    auto x = make_var(xRaw);
    auto yRaw = ArrayXX(1, 2);
    yRaw << 3, 4;
    auto y = make_var(yRaw);

//    auto x = make_var(Mat::Random(2, 2));
//    auto y = make_var(Mat::Random(2, 2));

//    auto expr = reduceSum(log(x*x + x*y + y*y)*x*y);
//    auto expr = x * y;
    auto expr = reduceSum(vvt(x, y) * vvt(x, y));
    auto forwardResult = expr->evalForward();

    std::cout << "x = \n" << x->evalForward() << std::endl;
    std::cout << "y = \n" << y->evalForward() << std::endl;

    std::cout << "f(x, y) = \n" << forwardResult << std::endl;
    expr->differentiateBackward();
    std::cout << "df(x,y)/dx = \n" << x->derivative() << std::endl;
    std::cout << "df(x,y)/dy = \n" << y->derivative() << std::endl;

    return 0;
}
