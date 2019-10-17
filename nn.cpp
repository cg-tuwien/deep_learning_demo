/*
 * Copyright (c) 2019, Adam Celarek | Research Unit of Computer Graphics | TU Wien
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of mosquitto nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "nn.h"

namespace  {
inline ExpressionPtr epsLike(ExpressionPtr mat) {
    return Constant::make(mat->rows(), mat->cols(), 0.00000001f);
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
ExpressionPtr nn::numerical_instable_softmax(ExpressionPtr mat)
{
    Q_ASSERT(mat->cols() == 1);
    auto exponentials = exp(mat);
    auto sum = onesLike(mat) * reduceSum(exponentials);
    return cwisediv(exponentials, sum);
}

ExpressionPtr nn::softmax(ExpressionPtr mat)
{
    Q_ASSERT(mat->cols() == 1);
    auto exponentials = normExp(mat);
    auto sum = onesLike(mat) * reduceSum(exponentials);
    return cwisediv(exponentials, sum);
}

ExpressionPtr nn::mse(ExpressionPtr a, ExpressionPtr b)
{
    auto diff = a - b;
    return reduceSum(cwisemul(diff, diff));
}

ExpressionPtr nn::crossEntropy(ExpressionPtr pred, ExpressionPtr truth)
{
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
