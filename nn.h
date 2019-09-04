#ifndef NNTOOLS_H
#define NNTOOLS_H

#include "Expression.h"

namespace nn {

ExpressionPtr sigmoid(ExpressionPtr mat);
ExpressionPtr softplus(ExpressionPtr mat);
ExpressionPtr mse(ExpressionPtr a, ExpressionPtr b);

}

#endif // NNTOOLS_H
