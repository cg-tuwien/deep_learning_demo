#include <iostream>
#include "Eigen/Core"
#include "Expression.h"
#include "nn.h"

int main(int argc, char *argv[])
{
//    auto x = Variable::make(2, 4);
//    x->value() << 0, 0, 1, 1,
//                  0, 1, 0, 1;

//    auto y = Variable::make(1, 4);
//    y->value() << 0, 0, 0, 1;

    auto W = Variable::make(ArrayXX::Random(2, 2));

    auto x = Variable::make(Eigen::ArrayXf::LinSpaced(20, -5, 5));
    auto y = Variable::make(20, 1, 1);

//    auto x = make_var(Mat::Random(2, 2));
//    auto y = make_var(Mat::Random(2, 2));

//    auto expr = reduceSum(log(x*x + x*y + y*y)*x*y);
//    auto expr = x * y;
//    auto expr = reduceProd(matmul(x, y) * matmul(x, y));

    auto expr = reduceSum(cwisemul(nn::softplus(x), y));

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
