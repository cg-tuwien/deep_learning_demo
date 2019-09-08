#include "Tests.h"
#include <iostream>

#include <QtGlobal>

#include "Expression.h"
#include "nn.h"

namespace {
void testErr(std::string condition, std::string file, int line) {
    std::cerr << "test \"" << condition << "\" failed in " << file << ", line " << line << std::endl;
    throw 0;
}
float sq(float v) { return v*v; }

#define TUW_CHECK(cond) ((!(cond)) ? testErr(#cond, __FILE__, __LINE__) : ((void)0))

void testSimpleDescent()
{
    std::cout << "testSimpleDescent()" << std::endl;
    float learningRate = 0.05f;
    auto offsetsX = Constant::make(ArrayXX::Random(10, 1) * 5);
    auto offsetsY = Constant::make(ArrayXX::Random(10, 1) * 5);
    auto x = Variable::make(ArrayXX::Random(10, 1) * 5);
    auto y = Variable::make(ArrayXX::Random(10, 1) * 5);

    auto yTr = (y + offsetsY);
    auto f = reduceSum(cwisemul((x + offsetsX), (x + offsetsX)) + cwisemul(yTr, yTr));
    for (int i = 0; i < 1000; ++i) {
        x->resetGradient();
        y->resetGradient();
        f->reset();
        f->evalForward();
        f->differentiateBackward();
        x->value() -= x->gradient() * learningRate;
        y->value() -= y->gradient() * learningRate;
//        std::cout << "f = " << f->evalForward() << std::endl;
//        std::cout << "x = " << x->value().transpose() << std::endl;
//        std::cout << "y = " << y->value().transpose() << std::endl;
    }
//    std::cout << "offsetsX = " << offsetsX->value().transpose() << std::endl;
//    std::cout << "offsetsY = " << offsetsY->value().transpose() << std::endl;
    TUW_CHECK(f->evalForward()(0) < 0.000001f);
    TUW_CHECK((x->value() + offsetsX->value()).abs().sum() < 0.0001f);
    TUW_CHECK((y->value() + offsetsY->value()).abs().sum() < 0.0001f);
}

void testReluLayerMath()
{
    std::cout << "testReluLayerMath()" << std::endl;
    auto W = Variable::make(ArrayXX::Random(100, 10));
    auto y = Constant::make(ArrayXX::Random(10, 1));
    auto f = relu(W * y);
//    std::cout << "y = " << (y)->evalForward().transpose() << std::endl;
//    std::cout << "Wx = " << (W * y)->evalForward().transpose() << std::endl;
//    std::cout << "f = " << f->evalForward().transpose() << std::endl;
    auto fEval = f->evalForward();
    reduceSum(f)->differentiateBackward();
//    std::cout << "df/dW = " << W->gradient() << std::endl;

    // the gradient should be zero where W.row dot y < 0 and y.transpose otherwise
    auto WyGt0 = ((W->value().matrix() * y->value().matrix()).array() > 0).cast<float>() * 0.99 + 0.01; // 100x1 boolean, but we have leaky relu now, so..
    TUW_CHECK( (W->gradient() - (WyGt0.matrix() * y->value().matrix().transpose()).array()).abs().sum() < 0.0000001f);
//    TUW_CHECK((fEval - x->value().max(0)).abs().sum() < 0.00000001f);
    //    TUW_CHECK((x->gradient() - (x->value() > 0).cast<float>()).abs().sum() < 0.00000001f);
}

void testReluLayerNet()
{
    std::cout << "testReluLayerNet()" << std::endl;
    auto y = Constant::make(ArrayXX::Random(10, 1));
    auto layer = nn::Layer::make(y, 10, relu);
    auto f = layer->out;

    auto fEval = f->evalForward();
    reduceSum(f)->differentiateBackward();

    auto W = layer->W->value();
    auto b = layer->b->value();
//    std::cout << "y = " << (y)->evalForward().transpose() << std::endl;
//    std::cout << "b = " << layer->b->evalForward().transpose() << std::endl;
//    std::cout << "Wx - b = " << (layer->W * y - layer->b)->evalForward().transpose() << std::endl;
//    std::cout << "f = " << f->evalForward().transpose() << std::endl;

//    std::cout << "df/dW = " << layer->W->gradient() << std::endl;
//    std::cout << "df/db = " << layer->b->gradient() << std::endl;

    auto WyGt0 = ((W.matrix() * y->value().matrix()).array() > b).cast<float>() * 0.99 + 0.01; // 100x1 boolean, but we have leaky relu now, so..
//    std::cout << "WyGt0 = " << WyGt0 << std::endl;
    TUW_CHECK( (layer->W->gradient() - (WyGt0.matrix() * y->value().matrix().transpose()).array()).abs().sum() < 0.0000001f);
    TUW_CHECK( (layer->b->gradient() + WyGt0).abs().sum() < 0.0000001f);

}

void testBinaryOpClass(const ArrayXX& yData)
{
    std::cout << "testBinaryOpClass(" << yData << ")" << std::endl;
    bool debug = false;
    auto xData = ArrayXX(2, 4);
    xData << 0, 0, 1, 1,
             0, 1, 0, 1;

    const int nEpochs = 1400;

    auto net = nn::Net::make(xData.col(0), yData.col(0), {4}, relu, nn::sigmoid, nn::crossEntropy2, 0.1f);
    float oldCost = 0;
    float runningCostDiffMean = 0;
    for (int i = 0; i < nEpochs; ++i) {
        if (debug)
            net->printWeights();
        net->resetGradient();
        float cost = 0;
        int good = 0;
        for (int j = 0; j < 4; ++j) {
            auto x = xData.col(j);
            auto y = yData.col(j);
            float loss = net->loss(x, y);
            cost += loss;
            net->costOutExpr->differentiateBackward();

            auto yPred = net->output(x);
            if (debug)
                std::cout << "in: " << x.transpose() << ", pred: " << yPred << ", truth: " << y << std::endl;
            good += ((y > 0.5f) == (yPred > 0.5f)).all() ;
        }
        net->applyGradient(debug);
        if (debug)
            std::cout << "epoch " << i << ", training cost = " << cost / 4.f << " percentage correct = " << float(good) / 4.f << std::endl;

        runningCostDiffMean = (runningCostDiffMean * 0.98f) + (cost - oldCost) * 0.02f;
        oldCost = cost;

        if (debug && runningCostDiffMean >= 0.001f) {
            std::cout << "runningCostDiffMean = " << runningCostDiffMean << ", i=" << i << std::endl;
        }

        if (i > 500) {
            TUW_CHECK(runningCostDiffMean < 0.001f);
        }

        if (i > 800) {
            TUW_CHECK(good == 4);
            TUW_CHECK(cost / 4.f < 0.2f);
        }
    }
}

template<typename ActivationFun, typename LossFunction>
void testLossGradients(ActivationFun activationFun, LossFunction lossFun) {
    std::cout << "testLossGradients()" << std::endl;
	float learningRate = 0.1f;
    auto x = Variable::make(ArrayXX::Random(10, 1) * 5);

    auto target = Variable::make(ArrayXX::Zero(10, 1));
    target->value()(0) = 1.f;

    auto pred = activationFun(x);
    auto loss = lossFun(pred, target);

    for (int i = 0; i < 1000; ++i) {
        x->resetGradient();
        loss->reset();
//        std::cout << "loss: " << loss->evalForward() << std::endl;
        loss->differentiateBackward();
//        std::cout << "targ: " << target->evalForward().transpose() << std::endl;
//        std::cout << "pred: " << pred->evalForward().transpose() << std::endl;
//        std::cout << "x: " << x->value().transpose() << std::endl;
//        std::cout << "x gradient: " << x->gradient().transpose() << std::endl;
        TUW_CHECK(std::abs(pred->evalForward().sum() - 1.f) < 0.001f);
        x->value() -= x->gradient() * learningRate;
//        std::cout << "f = " << f->evalForward() << std::endl;
//        std::cout << "x = " << x->value().transpose() << std::endl;
//        std::cout << "y = " << y->value().transpose() << std::endl;
	}
	TUW_CHECK(loss->evalForward()(0) < 0.01f);
	TUW_CHECK((pred->evalForward() - target->value()).abs().sum() < 0.02f);
	TUW_CHECK(x->gradient().abs().sum() < 0.02f);
}

void testSoftMax() {
    std::cout << "testSoftMax()" << std::endl;

	ArrayXX chainFactors = ArrayXX::Zero(10, 1);
    chainFactors(0, 0) = -1.f;

	auto x = Variable::make(ArrayXX::Random(10, 1) * 10);
//	std::cout << "x = " << x->value().transpose() << std::endl;

    auto softMax_instable = nn::numerical_instable_softmax(x);
    ArrayXX result_instable = softMax_instable->evalForward();
    softMax_instable->differentiateBackward(chainFactors);
    ArrayXX gradient_instable = x->gradient();
	x->resetGradient();

    auto softMax_new = nn::softmax(x);
	ArrayXX result_new = softMax_new->evalForward();
	softMax_new->differentiateBackward(chainFactors);
    ArrayXX gradient_new = x->gradient();
	x->resetGradient();

	TUW_CHECK((result_instable - result_new).abs().sum() < 0.001f);
	TUW_CHECK((gradient_instable - gradient_new).abs().sum() < 0.001f);
//	std::cout << "instable result  =\t" << result_instable.transpose() << std::endl;
//	std::cout << "new result       =\t" << result_new.transpose() << std::endl;

//	std::cout << "instable gradient =\t" << gradient_instable.transpose() << std::endl;
//	std::cout << "new gradient      =\t" << gradient_new.transpose() << std::endl;

}

}

void test()
{
	testSimpleDescent();
	testReluLayerMath();
	testReluLayerNet();

	auto yData = ArrayXX(1, 4);
	yData << 0, 0, 0, 0; // no
	testBinaryOpClass(yData);
	yData << 1, 1, 1, 1; // yes
	testBinaryOpClass(yData);
	yData << 0, 1, 1, 1; // or
	testBinaryOpClass(yData);
	yData << 0, 0, 0, 1; // and
	testBinaryOpClass(yData);
	yData << 0, 1, 1, 0; // xor
	testBinaryOpClass(yData);

	testLossGradients(nn::numerical_instable_softmax, nn::crossEntropy);
	testLossGradients(nn::numerical_instable_softmax, nn::crossEntropy2);

	testLossGradients(nn::softmax, nn::crossEntropy);
	testLossGradients(nn::softmax, nn::crossEntropy2);

    testSoftMax();
}
