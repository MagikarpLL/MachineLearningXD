# coding:utf-8
from me.tool import Data
import numpy as np
import operator


def getDataSet(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = np.array(data)
    labels = (ndata[:, 30:]).T
    result = ndata[:, :30]
    return result, labels[0]


def basic_KNN(input, trainData, labels, k):
    trainDataSize = trainData.shape[0]
    # 计算距离
    diffMat = np.tile(input, (trainDataSize, 1)) - trainData
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def KNN_Test(trainFile, testFile, k=3):
    trainData, trainLabels = getDataSet(trainFile)
    testData, testLabels = getDataSet(testFile)
    numTestData = testData.shape[0]
    errorCount = 0.0
    fnCount = 0
    tpCount = 0
    for i in range(numTestData):
        classifierResult = basic_KNN(testData[i], trainData, trainLabels, k)
        if (classifierResult != testLabels[i]):
            errorCount += 1.0
            if testLabels[i] == -1:
                fnCount += 1
        else:
            if testLabels[i] == -1:
                tpCount += 1
    print "一共测试 %d 条数据， 错误 %d 条数据， 误判 %d 条数据 ， 错误率为 %f, 误判率为%f" % (
        numTestData, errorCount, fnCount, (errorCount / float(numTestData)), (fnCount / float(tpCount + fnCount)))


def mainFunction(k):
    KNN_Test('train_500_1.txt', 'test_500_1.txt', k)
    KNN_Test('train_500_2.txt', 'test_500_2.txt', k)
    KNN_Test('train_500_3.txt', 'test_500_3.txt', k)
    KNN_Test('train_500_4.txt', 'test_500_4.txt', k)
    KNN_Test('train_1000_1.txt', 'test_1000_1.txt', k)
    KNN_Test('train_1000_2.txt', 'test_1000_2.txt', k)
    KNN_Test('train_1000_3.txt', 'test_1000_3.txt', k)
    KNN_Test('train_1000_4.txt', 'test_1000_4.txt', k)
    KNN_Test('train_2000_1.txt', 'test_2000_1.txt', k)
    KNN_Test('train_2000_2.txt', 'test_2000_2.txt', k)
    KNN_Test('train_2000_3.txt', 'test_2000_3.txt', k)
    KNN_Test('train_2000_4.txt', 'test_2000_4.txt', k)
    KNN_Test('train_3000_1.txt', 'test_3000_1.txt', k)
    KNN_Test('train_3000_2.txt', 'test_3000_2.txt', k)
    KNN_Test('train_3000_3.txt', 'test_3000_3.txt', k)
    KNN_Test('train_3000_4.txt', 'test_3000_4.txt', k)
    return


mainFunction(25)
