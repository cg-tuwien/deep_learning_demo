#include "nn.h"

namespace  {
inline ExpressionPtr onesLike(ExpressionPtr mat) {
    return Constant::make(mat->rows(), mat->cols(), 1);
}
inline ExpressionPtr zerosLike(ExpressionPtr mat) {
    return Constant::make(mat->rows(), mat->cols(), 0);
}
}

ExpressionPtr nn::sigmoid(ExpressionPtr mat)
{
    auto ones = onesLike(mat);
    return cwisediv(ones, (ones + exp(zerosLike(mat) - mat)));
}

ExpressionPtr nn::softplus(ExpressionPtr mat)
{
    return log(onesLike(mat) + exp(mat));
}

ExpressionPtr nn::mse(ExpressionPtr a, ExpressionPtr b)
{
    auto diff = a - b;
    return reduceSum(cwisemul(diff, diff));
}
