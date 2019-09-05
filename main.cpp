#include <iostream>
#include "Eigen/Core"
#include "Expression.h"
#include "nn.h"

int main(int argc, char *argv[])
{
    auto x = Variable::make(2, 4);
    x->value() << 0, 0, 1, 1,
                  0, 1, 0, 1;

    auto y = Variable::make(1, 4);
    y->value() << 1, 1, 1, 0;


    const float learningRate = 0.5f;
    const int nEpochs = 100;
    auto net = nn::Net::make(x, y, {3, 1}, nn::softplus, nn::mse, learningRate);

    auto netOut = net->output(x->value());

    std::cout << "x = \n" << x->evalForward() << std::endl;
    std::cout << "y = \n" << netOut << std::endl;

    for (int i = 0; i < nEpochs; ++i) {
        std::cout << "error = " << net->learnBatch(x->value(), y->value()) << "\ty = " << net->output(x->value()) << std::endl;
    }





    return 0;
}
