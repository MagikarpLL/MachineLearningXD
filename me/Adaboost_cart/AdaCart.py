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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def getDataAndLabel(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = array(data)
    labels = (ndata[:, 30:]).T
    result = ndata[:, :30]
    return result, labels[0]


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


def chooseBestFeatureToSplit2(dataset, weights):
    weights = weights / sum(weights)

    dataSet = mat(dataset)

    xTr = dataSet[:, :30]
    yTr = dataSet[:, 30:]
    N, D = xTr.shape

    loss = zeros((N - 1, D))
    Q = dot(weights, yTr ** 2)
    feature = 0
    bestloss = Q
    cut = 0

    for d in range(D):
        x = xTr[:, d].flatten()
        idx = argsort(x)
        x = x[idx]
        w = weights[idx]
        y = yTr[idx]

        W_L = 0.
        W_R = 1.
        P_L = 0.
        P_R = dot(weights, yTr)

        for k in range(N - 1):
            W_L = W_L + w[k]
            W_R = W_R - w[k]
            P_L = P_L + w[k] * y[k]
            P_R = P_R - w[k] * y[k]
            if x[k] == x[k + 1]:
                continue
            else:
                loss = Q - P_L ** 2 / W_L - P_R ** 2 / W_R
                if loss < bestloss:
                    bestloss = loss
                    feature = d
                    cut = x[k]
    return feature, cut


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
def createTree(dataSet, weights, depth=3):
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
        splitDataSet(dataSet, bestFeat, bestValue, 'lt'), weights, depth - 1)
    myTree[bestFeatLabel]['>' + str(round(float(bestValue), 3))] = createTree(
        splitDataSet(dataSet, bestFeat, bestValue, 'gt'), weights, depth - 1)
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
    time_start = time.time()

    min_weights = weights.min()
    newDataSet = list(dataSet)

    # newDataSet = []
    # 最小权重样本为1,权重大的样本对应重复
    # for i in range(len(dataSet)):
    #    newDataSet.extend([dataSet[i]] * int(math.ceil(float(array(weights.T)[0][i] / min_weights))))
    bestWeekClass = {}
    dataMatrix = mat(dataSet)
    m, n = shape(dataMatrix)
    bestClassEst = mat(zeros((m, 1)))

    weekCartTree = createTree(newDataSet, weights, depth)

    errorList, predictResult = cartClassify(dataSet, weekCartTree)
    weightedError = weights.T * errorList
    bestWeekClass['cart'] = weekCartTree
    return bestWeekClass, predictResult, weightedError


def CartAdaboostTrain(dataSet, num=1, depth=3):
    weekCartClassList = []
    classList = mat([int(example[-1]) for example in dataSet])
    m = len(dataSet)
    weights = mat(ones((m, 1)) / m)
    finallyPredictResult = mat(zeros((m, 1)))
    for i in range(num):
        bestWeekClass, bestPredictValue, error = weekCartClass(dataSet, weights, depth)
        alpha = -float(0.5 * log(1 - error) / max(error, 1e-16))
        bestWeekClass['alpha'] = alpha
        expon = multiply(-1 * alpha * mat(classList).T, bestPredictValue)
        weights = multiply(weights, exp(expon))
        weights = weights / weights.sum()
        finallyPredictResult += alpha * mat(bestPredictValue)
        nowPredictError = multiply(sign(finallyPredictResult) != mat(classList).T, ones((m, 1)))
        errorRate = nowPredictError.sum() / m
        print "depth_" + str(depth) + '_num_' + str(num) + "_第" + str(i) + '个完成'
        bestWeekClass['error_rate'] = errorRate
        weekCartClassList.append(bestWeekClass)
        if errorRate == 0.0: break
    return weekCartClassList, finallyPredictResult


def getData(fileName):
    data = Data.getDataFromFile(fileName)
    return data


def getFeatureName():
    featName = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
                'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon',
                'port',
                'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
                'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow',
                'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index',
                'Links_pointing_to_page', 'Statistical_report', 'Result', ]
    return featName


def storeTree(inputTree, fileName):
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


'''
def trainAdaCart(trainData, storeName, depth, num):
    dataSet = getData(trainData)
    weekCartClass, finallyPredictResult = CartAdaboostTrain(dataSet, num, depth)
    storeName = 'depth' + str(depth) + '_' + str(num) + '/' + storeName
    storeTree(weekCartClass, storeName)
    print 'depth: ' + str(depth) + '____' + 'num: ' + str(num) + '____' + 'complete'
    print finallyPredictResult.T
'''


def trainAdaCart(trainData, testData, depth, num):
    trainData, trainLabel = getDataAndLabel(trainData)
    testData, testLabel = getDataAndLabel(testData)

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth, min_samples_split=20, min_samples_leaf=5),
                             algorithm='SAMME',
                             n_estimators=num, learning_rate=0.8)
    bdt.fit(trainData, trainLabel)
    result = bdt.predict(testData)

    errorNum = 0
    for i in range(len(testLabel)):
        if testLabel[i] != result[i]:
            errorNum += 1

    print errorNum / float(len(testLabel))

    return


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


def testAdaCart(trainData, treeFile):
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
    errorRate = float(errorNum) / len(testData)
    print errorRate
    return errorRate


def trainAllFun(depth):
    trainAdaCart('train_3000_1.txt', 'test_3000_1.txt', depth, 10)
    trainAdaCart('train_3000_2.txt', 'test_3000_2.txt', depth, 10)
    trainAdaCart('train_3000_3.txt', 'test_3000_3.txt', depth, 10)
    trainAdaCart('train_3000_4.txt', 'test_3000_4.txt', depth, 10)

    trainAdaCart('train_3000_1.txt', 'test_3000_1.txt', depth, 30)
    trainAdaCart('train_3000_2.txt', 'test_3000_2.txt', depth, 30)
    trainAdaCart('train_3000_3.txt', 'test_3000_3.txt', depth, 30)
    trainAdaCart('train_3000_4.txt', 'test_3000_4.txt', depth, 30)

    trainAdaCart('train_3000_1.txt', 'test_3000_1.txt', depth, 60)
    trainAdaCart('train_3000_2.txt', 'test_3000_2.txt', depth, 60)
    trainAdaCart('train_3000_3.txt', 'test_3000_3.txt', depth, 60)
    trainAdaCart('train_3000_4.txt', 'test_3000_4.txt', depth, 60)

    trainAdaCart('train_3000_1.txt', 'test_3000_1.txt', depth, 120)
    trainAdaCart('train_3000_2.txt', 'test_3000_2.txt', depth, 120)
    trainAdaCart('train_3000_3.txt', 'test_3000_3.txt', depth, 120)
    trainAdaCart('train_3000_4.txt', 'test_3000_4.txt', depth, 120)
    return


def AdaCartMain():
    # trainAdaCart('train_500_1.txt', 'test.txts' , 3, 10)
    # error = testAdaCart(trainFile, storeFile)

    trainAllFun(3)
    trainAllFun(5)
    trainAllFun(8)
    trainAllFun(10)
    return


AdaCartMain()
