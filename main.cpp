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
    y->value() << 0, 1, 1, 0;

    auto W1 = Variable::make(ArrayXX::Random(3, 2));
    auto b1 = Variable::make(ArrayXX::Random(3, 1));
    auto W2 = Variable::make(ArrayXX::Random(1, 3));
    auto b2 = Variable::make(ArrayXX::Random(1, 1));

//    auto net = nn::softplus(W1 * x - b1 * Constant::make(b1->rows(), x->cols(), 1));
    auto layer1 = nn::neuron(W1, b1, x, nn::softplus);
    auto layer2 = nn::neuron(W2, b2, layer1, nn::softplus);
    auto net = layer2;
    auto cost = nn::mse(net, y);

    const float learningRate = 0.1f;
    const int nEpochs = 10000;
    std::cout << "x = \n" << x->evalForward() << std::endl;
    std::cout << "y = \n" << net->evalForward() << std::endl;
    for (int i = 0; i < nEpochs; ++i) {
        auto forwardResult = cost->evalForward();
        std::cout << "error = \n" << forwardResult << std::endl;
        std::cout << "y = \n" << net->evalForward() << std::endl;
//        std::cout << "W = \n" << W1->evalForward() << std::endl;


        cost->differentiateBackward();
//        std::cout << "df/dW = \n" << W1->derivative() << std::endl;
        W1->value() -= W1->derivative() * learningRate;
        b1->value() -= b1->derivative() * learningRate;
        W2->value() -= W2->derivative() * learningRate;
        b2->value() -= b2->derivative() * learningRate;

        W1->resetDerivative();
        b1->resetDerivative();
        W2->resetDerivative();
        b2->resetDerivative();

        cost->reset();
        std::cout << "=========================================" << std::endl;
    }





    return 0;
}
