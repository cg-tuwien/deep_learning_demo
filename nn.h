#ifndef NNTOOLS_H
#define NNTOOLS_H

#include <iostream>
#include <vector>

#include <QtGlobal>
#include <QtWidgets/QLabel>

#include <random>

#include "Expression.h"

namespace nn {

ExpressionPtr sigmoid(ExpressionPtr mat);
ExpressionPtr softplus(ExpressionPtr mat);
ExpressionPtr numerical_instable_softmax(ExpressionPtr mat);
ExpressionPtr softmax(ExpressionPtr mat);
ExpressionPtr mse(ExpressionPtr a, ExpressionPtr b);
ExpressionPtr crossEntropy(ExpressionPtr pred, ExpressionPtr truth);
ExpressionPtr crossEntropy2(ExpressionPtr pred, ExpressionPtr truth);

struct Layer;
using LayerPtr = std::shared_ptr<Layer>;

struct Layer {
    Layer() = default;
    Layer(const Layer&) = delete;
    VariablePtr W;
    VariablePtr b;
    ExpressionPtr out;

    template<typename Function>
    static LayerPtr make(ExpressionPtr input, int nNeurons,  Function activationFun) {
        static std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.f, 0.2f);
        auto normal = [&] (int) {return distribution(generator);};

    //    VariablePtr W, VariablePtr b
        auto W = Variable::make(ArrayXX::NullaryExpr(nNeurons, input->rows(), normal));
        auto b = Variable::make(nNeurons, 1, 0.f);

//		std::cout << "W:\n" << W->value().transpose() << std::endl;
//		std::cout << "b:" << b->value().transpose() << std::endl;

        LayerPtr l = std::make_shared<Layer>();
        l->W = W;
        l->b = b;
		l->out = activationFun(W * input - b * Constant::make(1, input->cols(), 1));

        return l;
    }

    float applyGradient(float learningRate) {
        W->value() -= W->gradient() * learningRate;
        b->value() -= b->gradient() * learningRate;
        return W->gradient().abs().mean() * learningRate + b->gradient().abs().mean() * learningRate;
    }
    void resetGradient() {
        W->resetGradient();
        b->resetGradient();
    }
};

struct Net;
using NetPtr = std::shared_ptr<Net>;
struct Net {
    Net() = default;
    Net(const Net&) = delete;

    std::vector<LayerPtr> layers;
    ConstantPtr input;
    ConstantPtr target;
    ExpressionPtr outExpr;
    ExpressionPtr costOutExpr;
    float learningRate;

    template<typename ActivationFunction, typename ClassificationFunction, typename CostFunction>
    static NetPtr make(const ArrayXX& input, const ArrayXX& target, const std::vector<int>& layers,
                       ActivationFunction activationFun,  ClassificationFunction classificationFun, CostFunction costFun, float learningRate) {
        NetPtr net = std::make_shared<Net>();
        net->input = Constant::make(input);
        net->target = Constant::make(target);

        ExpressionPtr layerInput = net->input;
        for (auto nNeurons : layers) {
            net->layers.push_back(Layer::make(layerInput, nNeurons, activationFun));
            layerInput = net->layers.back()->out;
        }
        net->layers.push_back(Layer::make(layerInput, int(target.rows()), classificationFun));
        net->outExpr = net->layers.back()->out;
        net->costOutExpr = costFun(net->outExpr, net->target);
        net->learningRate = learningRate;
        return net;
    }

    ArrayXX output(const ArrayXX& inputData) const {
        Q_ASSERT(input->rows() == inputData.rows());
        input->value() = inputData;
        outExpr->reset();
        return outExpr->evalForward();
    }

    float loss(const ArrayXX& inputData, const ArrayXX& targetData) {
        Q_ASSERT(input->rows() == inputData.rows());
        Q_ASSERT(target->rows() == targetData.rows());
        Q_ASSERT(input->cols() == inputData.cols());
        Q_ASSERT(target->cols() == targetData.cols());

        input->value() = inputData;
        target->value() = targetData;

        costOutExpr->reset();
        return  costOutExpr->evalForward()(0);
    }

    void applyGradient(bool debug_out = false)
    {
        int idx = 0;
        for (const auto& layer : layers) {
            float avgStepLength = layer->applyGradient(learningRate);
            if (debug_out)
                std::cout << "layer " << idx << ", average step length = " << avgStepLength << std::endl;
            ++idx;
        }
        resetGradient();
    }

    void resetGradient()
    {
        for (const auto& layer : layers) {
            layer->resetGradient();
        }
    }

    float learn(const ArrayXX& inputData, const ArrayXX& targetData) {
//        printWeights();
        resetGradient();
        auto error = loss(inputData, targetData);
        costOutExpr->differentiateBackward();

        applyGradient();
        printWeights();

        return error;
    }

    void printWeights() const {
        int idx = 0;
        static bool print = true;
        if (!print)
            return;
        for (const auto& layer : layers) {
            std::cout << "layer " << idx << ": W: "
                      << layer->W->value().transpose() << "\n"
                      << "   b: " << layer->b->value().transpose() << std::endl;
            ++idx;
        }
    }
};


}

#endif // NNTOOLS_H
