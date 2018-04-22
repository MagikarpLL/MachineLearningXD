# coding: utf-8
import sys

sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
import pickle
from itertools import *
import operator, time, math
import matplotlib.pyplot as plt
from numpy import *
from treePlotter import createPlot



# 计算一个数据集的gini系数
def calGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1
    for label in labelCounts.keys():
        prop = float(labelCounts[label]) / numEntries
        gini -= prop * prop
    return gini


# 根据对应值划分数据集
def splitDataSet(dataSet, axis, value, threshold):
    retDataSet = []
    if threshold == 'lt':
        for featVec in dataSet:
            if featVec[axis] <= value:
                retDataSet.append(featVec)
    else:
        for featVec in dataSet:
            if featVec[axis] > value:
                retDataSet.append(featVec)
    return retDataSet


# 返回最好的特征以及特征值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestGiniGain = 1.0
    bestFeature = -1
    bestValue = ""
    # 遍历特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = list(set(featList))
        uniqueVals.sort()
        for value in uniqueVals:
            GiniGain = 0.0
            # 左增益
            left_subDataSet = splitDataSet(dataSet, i, value, 'lt')
            left_prob = len(left_subDataSet) / float(len(dataSet))
            GiniGain += left_prob * calGini(left_subDataSet)
            # 右增益
            right_subDataSet = splitDataSet(dataSet, i, value, 'gt')
            right_prob = len(right_subDataSet) / float(len(dataSet))
            GiniGain += right_prob * calGini(right_subDataSet)
            # 与最好的结果比较并记录
            if (GiniGain < bestGiniGain):
                bestGiniGain = GiniGain
                bestFeature = i
                bestValue = value
    return bestFeature, bestValue


# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


# 生成一棵指定深度的cart树
def createTree(dataSet, depth=3):
    classList = [example[-1] for example in dataSet]
    # 如果到达了指定深度，直接多数表决决定结果
    if depth == 0:
        return majorityCnt(classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet) == 1:
        return majorityCnt(classList)
    bestFeat, bestValue = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = str(bestFeat) + ":" + str(bestValue)
    if bestFeat == -1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    myTree[bestFeatLabel]['<=' + str(round(float(bestValue), 3))] = createTree(
        splitDataSet(dataSet, bestFeat, bestValue, 'lt'), depth - 1)
    myTree[bestFeatLabel]['>' + str(round(float(bestValue), 3))] = createTree(
        splitDataSet(dataSet, bestFeat, bestValue, 'gt'), depth - 1)
    return myTree


# 将CART树的节点名翻译为特征名
def translateTree(tree, labels):
    if type(tree) is not dict:
        return tree
    root = tree.keys()[0]
    feature, threshold = root.split(":")
    feature = int(feature)
    myTree = {labels[feature]: {}}
    for key in tree[root].keys():
        myTree[labels[feature]][key] = translateTree(tree[root][key], labels)
    return myTree


def predict(tree, sample):
    if type(tree) is not dict:
        return tree
    root = tree.keys()[0]
    feature, threshold = root.split(":")
    feature = int(feature)
    threshold = float(threshold)
    if sample[0][feature] > threshold:
        return predict(tree[root]['>' + str(round(float(threshold), 3))], sample)
    else:
        return predict(tree[root]['<=' + str(round(float(threshold), 3))], sample)


# 通过CART树来预测数据集
def cartClassify(dataSet, tree):
    dataMatrix = mat(dataSet)
    # 返回预测对还是错，对为0,错为1
    errorList = ones((shape(dataMatrix)[0], 1))
    predictResult = []
    classList = [example[-1] for example in dataSet]
    for i in range(len(dataMatrix)):
        res = predict(tree, dataMatrix[i].getA())
        errorList[i] = (res != classList[i])
        predictResult.append([int(res)])
    return errorList, predictResult


# 训练弱分类器，通过调整样本的个数来达到调整样本权重的目的
def weekCartClass(dataSet, weights, depth=3):
    min_weights = weights.min()
    newDataSet = []
    # 最小权重样本为1,权重大的样本对应重复
    for i in range(len(dataSet)):
        newDataSet.extend([dataSet[i]] * int(math.ceil(float(array(weights.T)[0][i] / min_weights))))
    bestWeekClass = {}
    dataMatrix = mat(dataSet)
    m,n = shape(dataMatrix)
    bestClassEst = mat(zeros((m,1)))
    weekCartTree = createTree(newDataSet,depth)
    errorList, predictResult = cartClassify(dataSet, weekCartTree)
    weightedError = weights.T * errorList
    bestWeekClass['cart'] = weekCartTree
    return bestWeekClass, predictResult, weightedError

def CartAdaboostTrain(dataSet, num=1, depth=3):
    weekCartClassList = []
    classList = mat([int(example[-1]) for example in dataSet])
    m = len(dataSet)
    weights = mat(ones((m,1))/m)
    finallyPredictResult = mat(zeros((m,1)))
    for i in range(num):
        bestWeekClass, bestPredictValue, error = weekCartClass(dataSet, weights, depth)
        alpha = -float(0.5 * log(1-error)/max(error, 1e-16))
        bestWeekClass['alpha'] = alpha
        expon = multiply(-1*alpha*mat(classList).T, bestPredictValue)
        weights = multiply(weights, exp(expon))
        weights = weights/weights.sum()
        finallyPredictResult += alpha*mat(bestPredictValue)
        nowPredictError = multiply(sign(finallyPredictResult) != mat(classList).T, ones((m,1)))
        errorRate = nowPredictError.sum()/m
        print "total error: ", errorRate
        bestWeekClass['error_rate'] = errorRate
        weekCartClassList.append(bestWeekClass)
        if errorRate == 0.0: break
    return weekCartClassList, finallyPredictResult

def getData(fileName):
    data = Data.getDataFromFile(fileName)
    return data

def getFeatureName():
    featName = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port',
                'HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow',
                'Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result',]
    return featName

def storeTree(inputTree, fileName):
    fw = open(fileName,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

def trainAdaCart(trainData, storeName, depth, num):
    dataSet = getData(trainData)
    weekCartClass, finallyPredictResult = CartAdaboostTrain(dataSet, num, depth)
    storeTree(weekCartClass, storeName)
    print finallyPredictResult.T


def predictTest(tree, sample):
    if type(tree) is not dict:
        return tree
    root = tree.keys()[0]
    feature, threshold = root.split(":")
    feature = int(feature)
    threshold = float(threshold)
    if sample[feature] > threshold:
        return predictTest(tree[root]['>' + str(round(float(threshold), 3))], sample)
    else:
        return predictTest(tree[root]['<=' + str(round(float(threshold), 3))], sample)

def testAdaCart(trainData,treeFile):
    tree = grabTree(treeFile)
    testData = getData(trainData)
    errorNum = 0
    for i in range(len(testData)):
        res = 0
        for j in range(len(tree)):
            res += tree[j]['alpha'] * predictTest(tree[j]['cart'], testData[i])


        res = sign(res)
        if res != testData[i][-1]:
            errorNum += 1
    errorRate = float(errorNum)/len(testData)
    print errorRate
    return errorRate


def AdaCartMain(trainFile, storeFile, depth, num):
    #trainAdaCart(trainFile, storeFile, depth, num)
    error = testAdaCart(trainFile, storeFile)
    return

AdaCartMain('test_50.txt', 'AdaCart_1000.txt', 3, 10)


