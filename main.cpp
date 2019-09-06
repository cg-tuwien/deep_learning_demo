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
    test();
    return 0;
    auto trainingList = getData("/home/madam/Downloads/mnist_png/training");
    std::random_shuffle(trainingList.begin(), trainingList.end());

    auto testList = getData("/home/madam/Downloads/mnist_png/testing");
    std::random_shuffle(testList.begin(), testList.end());

    std::cout << "training size = " << trainingList.size() << "  validation size = " << testList.size() << std::endl;

    Q_ASSERT(trainingList.size());
    Q_ASSERT(testList.size());

    const int nEpochs = 100;
    const float learningRate = 64.f;
    const int batchSize =  32;
    auto net = nn::Net::make(ArrayXX(28 * 28, 1), ArrayXX(10, 1), {64, 64}, relu, nn::softmax, nn::crossEntropy2, learningRate / batchSize);

    for (int i = 0; i < nEpochs; ++i) {
//        for (const auto& dataPair : trainingList) {
        for (size_t i = 0; i < trainingList.size(); ++i) {
            net->resetGradient();
            auto batchEnd = std::min(i + 1000, trainingList.size());
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
