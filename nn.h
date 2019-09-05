#ifndef NNTOOLS_H
#define NNTOOLS_H

#include <iostream>
#include <vector>

#include <QtGlobal>

#include "Expression.h"

namespace nn {

ExpressionPtr sigmoid(ExpressionPtr mat);
ExpressionPtr softplus(ExpressionPtr mat);
ExpressionPtr mse(ExpressionPtr a, ExpressionPtr b);

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
    //    VariablePtr W, VariablePtr b
        auto W = Variable::make(ArrayXX::Random(nNeurons, input->rows()));
        auto b = Variable::make(ArrayXX::Random(nNeurons, 1));

        LayerPtr l = std::make_shared<Layer>();
        l->W = W;
        l->b = b;
        l->out = activationFun(W * input - b * Constant::make(1, input->cols(), 1));

        return l;
    }

    void applyGradient(float learningRate) {
        W->value() -= W->derivative() * learningRate;
        b->value() -= b->derivative() * learningRate;

        W->resetDerivative();
        b->resetDerivative();
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
    ExpressionPtr costOut;
    float learningRate;

    template<typename ActivationFunction, typename CostFunction>
    static NetPtr make(ConstantPtr input, ConstantPtr target, const std::vector<int>& layers,  ActivationFunction activationFun, CostFunction costFun, float learningRate) {
        Q_ASSERT(layers.size());
        Q_ASSERT(layers.back() == target->rows());

        NetPtr net = std::make_shared<Net>();
        net->input = input;
        net->target = target;

        ExpressionPtr layerInput = input;
        for (auto nNeurons : layers) {
            net->layers.push_back(Layer::make(layerInput, nNeurons, activationFun));
            layerInput = net->layers.back()->out;
        }
        net->costOut = costFun(net->layers.back()->out, target);
        net->learningRate = learningRate;
        return net;
    }

    ArrayXX output(const ArrayXX& inputData) const {
        Q_ASSERT(input->rows() == inputData.rows());
        input->value() = inputData;
        return layers.back()->out->evalForward();
    }

    float learnBatch(const ArrayXX& inputData, const ArrayXX& targetData) {
        Q_ASSERT(input->rows() == inputData.rows());
        Q_ASSERT(target->rows() == targetData.rows());

        input->value() = inputData;
        target->value() = targetData;

        costOut->reset();
        auto error = costOut->evalForward()(0);

        costOut->differentiateBackward();
        applyGradient(learningRate);

        return error;
    }

    void applyGradient(float learningRate) {
        for (const auto& layer : layers) {
            layer->applyGradient(learningRate);
        }
    }
};


}

#endif // NNTOOLS_H
