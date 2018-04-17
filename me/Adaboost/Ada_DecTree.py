# coding: utf-8
import sys

sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
from numpy import *
import operator
from math import log
import treePlotter
import pickle

#多数投票决定该节点类型
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 计算数据集的香农熵
def calcShannonEnt(dataSet, classLabels):
    numEntries = len(dataSet)
    labelCounts = {}
    for i in range(numEntries):
        currentLabel = classLabels[i]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if(featVec[axis] == value):
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet,classLabels,D):
    numFeatures = len(dataSet[0])
    baseEntropy = calcShannonEnt(dataSet, classLabels)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)



    return bestFeature

def createTree(trainData, classLabels ,featName, D):
    #当类别完全相同时，停止继续划分
    if(classLabels.count(classLabels[0]) == len(classLabels)):
        return classLabels[0]
    #遍历完所有时，若分类还未结束，则返回出现次数最多的作为该节点类型
    if len(trainData[0]) == 0:
        return majorityCnt(classLabels)
    bestFeat = chooseBestFeatureToSplit(trainData,classLabels, D)


    return decTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def buildDecTree(trainData, classLabels, featName, D):
    decTree = createTree(trainData, classLabels ,featName, D)
    classEst = []
    dataNum = len(trainData)
    errorNum = 0
    for i in range(dataNum):
        testResult = classify(decTree, featName, trainData[i])
        classEst.append(testResult)
        if int(testResult) != int(classLabels[i]):
            errorNum += 1
    error = errorNum/float(dataNum)
    return decTree, error, classEst


def adaBoostTrainDT(trainData, classLabels, featName, numIt=40):
    weakClassArr = []
    m = shape(trainData)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        decTree, error, classEst = buildDecTree(trainData, classLabels, featName, D)
        print "D:", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        decTree['alpha'] = alpha
        weakClassArr.append(decTree)
        print "classEst:", classEst.T
        # 为下一次迭代计算D
        # 此处的multiply为对应元素相乘
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        # exp(x)返回x的指数
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 错误率累加计算
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        if errorRate <= 0.13: break
    return weakClassArr
