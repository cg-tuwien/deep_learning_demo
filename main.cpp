#include <iostream>
#include <vector>
#include <cmath>

#include "Eigen/Core"
#include "Expression.h"
#include "nn.h"

#include <QImage>
#include <QDir>

std::vector<std::pair<ArrayXX, QString>> getData(QString path) {
    std::vector<std::pair<ArrayXX, QString>> data;
    for (int i = 0; i < 10; ++i) {
        auto dataDir = QDir(QString("%1/%2/").arg(path).arg(i));
        auto entries = dataDir.entryInfoList({"*.png"});
        for (const auto& e : entries) {
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
        x(p, 0) = *(image.bits() + p) / 255.f;
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
    auto trainingList = getData("/home/madam/Downloads/mnist_png/training");
    std::random_shuffle(trainingList.begin(), trainingList.end());

    auto testList = getData("/home/madam/Downloads/mnist_png/testing");
    std::random_shuffle(testList.begin(), testList.end());

    std::cout << "training size = " << trainingList.size() << "  validation size = " << testList.size() << std::endl;

    Q_ASSERT(trainingList.size());
    Q_ASSERT(testList.size());

    const float learningRate = 0.1f;
    const int nEpochs = 100;
    auto net = nn::Net::make(ArrayXX(28 * 28, 1), ArrayXX(10, 1), {128, 64}, relu, nn::crossEntropy2, learningRate);

    for (int i = 0; i < nEpochs; ++i) {
        float err = 0;
        int good = 0;
        int counter = 0;
        for (const auto& dataPair : trainingList) {
            auto x = getImage(dataPair.second);
            auto y = dataPair.first;
            err += net->learn(x, y);

            auto yPred = net->output(x);
            good += number(y) == number(yPred);

            ++counter;
            if (counter % 5000 == 0) {
                std::cout << "training cost = " << err / trainingList.size() << " percentage correct: " << float(good) / 5000 << std::endl;
                good = 0;
//                std::cout << "counter = " << counter << " training cost = " << err / counter << std::endl;
            }
        }
        std::cout << "training cost = " << err / trainingList.size() << " percentage correct: " << float(good) / testList.size() << std::endl;

        err = 0;
        good = 0;
        for (const auto& dataPair : testList) {
            auto x = getImage(dataPair.second);
            auto y = dataPair.first;
            auto yPred = net->output(x);
            good += number(y) == number(yPred);
            err += net->cost(x, y);
        }
        std::cout << " test cost = " << err / testList.size() << " percentage correct: " << float(good) / testList.size() << std::endl;
    }


    return 0;
}
