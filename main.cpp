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

#include <iostream>
#include <vector>
#include <cmath>

#include <QImage>
#include <QDir>

#include "Eigen/Core"
#include "Expression.h"
#include "nn.h"
#include "Tests.h"


std::vector<std::pair<ArrayXX, QString>> getData(QString path) {
    std::vector<std::pair<ArrayXX, QString>> data;
	for (int i = 0; i < 10; ++i) {
        auto dataDir = QDir(QString("%1/%2/").arg(path).arg(i));
        auto entries = dataDir.entryInfoList({"*.png"});
        int cnt = 0;
        for (const auto& e : entries) {
//            if (cnt > 50) break;
            ++cnt;
            ArrayXX label = ArrayXX::Zero(10, 1);
            label(i, 0) = 1;
            QString dataPath = e.absoluteFilePath();
            data.push_back(std::make_pair(label, dataPath));
        }
    }
    return data;
}

ArrayXX getImage(QString path) {
    auto image = QImage(path).convertToFormat(QImage::Format_Grayscale8);
    ArrayXX x(28 * 28, 1);
    for (int p = 0; p < 28 * 28; ++p) {
        x(p, 0) = *(image.bits() + p) / 255.f - 0.5f;
    }
    return x;
}

int number(const ArrayXX& vec) {
    Eigen::Index idx;
    vec.col(0).maxCoeff(&idx);
    return int(idx);
}

int main(int argc, char *argv[])
{
//	test();
//	return 0;
    auto trainingList = getData("/home/madam/Downloads/mnist_png/training");
    std::random_shuffle(trainingList.begin(), trainingList.end());

    auto testList = getData("/home/madam/Downloads/mnist_png/testing");
    std::random_shuffle(testList.begin(), testList.end());

    std::cout << "training size = " << trainingList.size() << "  validation size = " << testList.size() << std::endl;

    Q_ASSERT(trainingList.size());
    Q_ASSERT(testList.size());

    const int nEpochs = 100;
	const float learningRate = 0.1f;
	const int batchSize =  2000;
	auto net = nn::Net::make(ArrayXX(28 * 28, 1), ArrayXX(10, 1), {64, 64}, relu, nn::softmax, nn::crossEntropy, learningRate / batchSize);

	for (int e = 0; e < nEpochs; ++e) {
//        for (const auto& dataPair : trainingList) {
        for (size_t i = 0; i < trainingList.size(); ++i) {
            net->resetGradient();
			auto batchEnd = std::min(i + batchSize, trainingList.size());
            float cost = 0;
            int good = 0;
            int counter = 0;
            for (; i < batchEnd; ++i) {
                auto x = getImage(trainingList.at(i).second);
                auto y = trainingList.at(i).first;
                float loss = net->loss(x, y);
                cost += loss;
                net->costOutExpr->differentiateBackward();

                auto yPred = net->output(x);
//				std::cout << "pred: " << yPred.transpose() << "\ntarget: " << y.transpose() << "loss: " << loss << std::endl;

				good += number(y) == number(yPred);
                ++counter;
            }
            net->applyGradient(true);
//            std::cout << "cost = " << cost << std::endl;
			std::cout << "training cost = " << cost / counter << " percentage correct: " << float(good) / counter << std::endl;
//                std::cout << "counter = " << counter << " training cost = " << err / counter << std::endl;
        }
//        std::cout << "training cost = " << err / trainingList.size() << " percentage correct: " << float(good) / testList.size() << std::endl;

        float cost = 0;
        int good = 0;
        for (const auto& dataPair : testList) {
            auto x = getImage(dataPair.second);
            auto y = dataPair.first;
            auto yPred = net->output(x);
            good += number(y) == number(yPred);
            cost += net->loss(x, y);
        }
        std::cout << "    test cost = " << cost / testList.size() << " percentage correct: " << float(good) / testList.size() << std::endl;
    }


    return 0;
}
