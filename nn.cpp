#include "nn.h"

namespace  {
inline ExpressionPtr epsLike(ExpressionPtr mat) {
    return Constant::make(mat->rows(), mat->cols(), 0.0001f);
}

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
    return cwisediv(ones, ones + exp(zerosLike(mat) - mat));
}

ExpressionPtr nn::softplus(ExpressionPtr mat)
{
    return log(onesLike(mat) + exp(mat));
}

// numerically instable
ExpressionPtr nn::softmax(ExpressionPtr mat)
{
    Q_ASSERT(mat->cols() == 1);
    auto exponentials = exp(mat);
    auto sum = onesLike(mat) * reduceSum(exponentials);
    return cwisediv(exponentials, sum + epsLike(sum));
}

ExpressionPtr nn::mse(ExpressionPtr a, ExpressionPtr b)
{
    auto diff = a - b;
    return reduceSum(cwisemul(diff, diff));
}

ExpressionPtr nn::crossEntropy(ExpressionPtr pred, ExpressionPtr truth)
{
    auto eps = Constant::make(truth->rows(), truth->cols(), 0.0001f);
    return Constant::make(1, 1, -1) * reduceSum(cwisemul(truth, log(pred + epsLike(pred))));
}

ExpressionPtr nn::crossEntropy2(ExpressionPtr pred, ExpressionPtr truth)
{
    auto ones = onesLike(truth);
    auto eps = epsLike(pred);
    return Constant::make(1, 1, -1) * reduceSum(cwisemul(truth, log(pred + eps)) + cwisemul((ones - truth), log(ones - pred + eps)));
}

//void nn::Descender::resetGradient()
//{
//    for (const auto& variable : variables) {
//        variable->resetGradient();
//    }
//}
