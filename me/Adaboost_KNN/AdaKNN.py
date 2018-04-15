#coding:utf-8
import sys
sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
import numpy as np
import operator

def getDataSet(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = np.array(data)
    labels = (ndata[:,30:]).T
    result = ndata[:,:30]
    return result,labels[0]

def basic_KNN(input, trainData, labels, k):
    trainDataSize = trainData.shape[0]
    #计算距离
    diffMat = np.tile(input, (trainDataSize,1)) - trainData
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def KNN_Test(trainFile, testFile):
    trainData, trainLabels = getDataSet(trainFile)
    testData, testLabels = getDataSet(testFile)
    numTestData = testData.shape[0]
    errorCount = 0.0
    for i in range(numTestData):
        classifierResult = basic_KNN(testData[i],trainData, trainLabels, 3)
        if(classifierResult != testLabels[i]):
            errorCount += 1.0
            print "预测结果为: %d, 实际结果为: %d" % (classifierResult, testLabels[i])
    print "一共测试 %d 条数据， 错误 %d 条数据， 错误率为 %f" %(numTestData, errorCount, (errorCount/float(numTestData)))





def mainFunction(trainFile, testFile):
    KNN_Test(trainFile,testFile)
    return

mainFunction('train_500.txt','test_100.txt')