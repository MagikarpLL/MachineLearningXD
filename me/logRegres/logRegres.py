# coding: utf-8
import sys

sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
import pickle
from numpy import *


# Logistic回归梯度上升优化算法
def loadDataSet(filename):
    path = '/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/data/current/'
    currPath = path + filename
    dataMat = []
    labelMat = []
    fr = open(currPath)
    for line in fr.readlines():
        nline = line.strip("\r\n")
        lineArr = [int(x) for x in nline.split(',')]
        tempList = []
        for i in range(30):
            temp = float(lineArr[i])
            tempList.append(temp)
        dataMat.append(tempList)
        if (int(lineArr[30]) == 1):
            labelMat.append(1)
        else:
            labelMat.append(0)
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 计算整个数据集的梯度，在数据集较大的时候，计算量会过于巨大
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    # weights = wei.getA()
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 对数据集中的每个样本，计算该样本的梯度，然后更新系数值
# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 前两个算法的收敛速度过慢，并且在大的波动停止后，还会出现小的周期性波动
# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones((1, n))
    weights = mat(weights)
    # weights = weights.T
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(multiply(dataMatrix[randIndex], weights)))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def store(input, fileName):
    fw = open(fileName, 'w')
    pickle.dump(input, fw)
    fw.close()


def grab(filename):
    fr = open(filename)
    return pickle.load(fr)


def trainFuc(trainFile, numIter, storeFile):
    dataSet, labelSet = loadDataSet(trainFile)
    weights = stocGradAscent1(mat(dataSet), labelSet, numIter)
    storeFile = '' + str(numIter) + '/' + storeFile
    store(weights, storeFile)

    print 'complete: ',storeFile

    # errorRate = classify(dataSet, labelSet, weights)
    # print errorRate
    return


def testFuc(testFile, numIter ,storeFile):
    storeFile = str(numIter) + '/' + storeFile
    weights = grab(storeFile)
    dataSet, labelSet = loadDataSet(testFile)
    allNum, errorNum, fnCount, tpCount = classify(dataSet, labelSet, weights)
    print '共测试 %d 条数据， 一共错误 %d 条数据， 错误率为 %f, 误判率为 %f' \
          %(allNum, errorNum, (errorNum/float(allNum)), (fnCount / float(tpCount + fnCount)))
    return


def classify(dataSet, labelSet, weights):
    errorNum = 0
    allNum = len(dataSet)
    fnCount = 0
    tpCount = 0
    for i in range(len(dataSet)):
        lineMat = mat(dataSet[i])
        temp = sum(multiply(lineMat, weights))
        result = sigmoid(temp)
        if (result > 0.5):
            result = 1
        else:
            result = 0
        if result != labelSet[i]:
            errorNum += 1
            if labelSet[i] == 0:
                fnCount += 1
        else:
            if labelSet[i] == 0:
                tpCount += 1
    return allNum, errorNum, fnCount, tpCount


def mainFuc(numIter):
    # testFuc(testFile, storeFile)

    testFuc('test_500_1.txt', numIter, 'train_500_1.txt')
    testFuc('test_500_2.txt', numIter, 'train_500_2.txt')
    testFuc('test_500_3.txt', numIter, 'train_500_3.txt')
    testFuc('test_500_4.txt', numIter, 'train_500_4.txt')
    testFuc('test_1000_1.txt', numIter, 'train_1000_1.txt')
    testFuc('test_1000_2.txt', numIter, 'train_1000_2.txt')
    testFuc('test_1000_3.txt', numIter, 'train_1000_3.txt')
    testFuc('test_1000_4.txt', numIter, 'train_1000_4.txt')
    testFuc('test_2000_1.txt', numIter, 'train_2000_1.txt')
    testFuc('test_2000_2.txt', numIter, 'train_2000_2.txt')
    testFuc('test_2000_3.txt', numIter, 'train_2000_3.txt')
    testFuc('test_2000_4.txt', numIter, 'train_2000_4.txt')
    testFuc('test_3000_1.txt', numIter, 'train_3000_1.txt')
    testFuc('test_3000_2.txt', numIter, 'train_3000_2.txt')
    testFuc('test_3000_3.txt', numIter, 'train_3000_3.txt')
    testFuc('test_3000_4.txt', numIter, 'train_3000_4.txt')
    return

'''
    trainFuc('train_500_1.txt', numIter, 'train_500_1.txt')
    trainFuc('train_500_2.txt', numIter, 'train_500_2.txt')
    trainFuc('train_500_3.txt', numIter, 'train_500_3.txt')
    trainFuc('train_500_4.txt', numIter, 'train_500_4.txt')
    trainFuc('train_1000_1.txt', numIter, 'train_1000_1.txt')
    trainFuc('train_1000_2.txt', numIter, 'train_1000_2.txt')
    trainFuc('train_1000_3.txt', numIter, 'train_1000_3.txt')
    trainFuc('train_1000_4.txt', numIter, 'train_1000_4.txt')
    trainFuc('train_2000_1.txt', numIter, 'train_2000_1.txt')
    trainFuc('train_2000_2.txt', numIter, 'train_2000_2.txt')
    trainFuc('train_2000_3.txt', numIter, 'train_2000_3.txt')
    trainFuc('train_2000_4.txt', numIter, 'train_2000_4.txt')
    trainFuc('train_3000_1.txt', numIter, 'train_3000_1.txt')
    trainFuc('train_3000_2.txt', numIter, 'train_3000_2.txt')
    trainFuc('train_3000_3.txt', numIter, 'train_3000_3.txt')
    trainFuc('train_3000_4.txt', numIter, 'train_3000_4.txt')
'''


mainFuc(1000)
