#ifndef NNTOOLS_H
#define NNTOOLS_H

#include <QtGlobal>

#include "Expression.h"

namespace nn {

ExpressionPtr sigmoid(ExpressionPtr mat);
ExpressionPtr softplus(ExpressionPtr mat);
ExpressionPtr mse(ExpressionPtr a, ExpressionPtr b);

template<typename Function>
ExpressionPtr neuron(VariablePtr W, VariablePtr b, ExpressionPtr input,  Function activationFun) {
    Q_ASSERT(W->rows() == b->rows());
    Q_ASSERT(W->cols() == input->rows());
    Q_ASSERT(b->cols() == 1);
    return activationFun(W * input - b * Constant::make(1, input->cols(), 1));
}

}

#endif // NNTOOLS_H
