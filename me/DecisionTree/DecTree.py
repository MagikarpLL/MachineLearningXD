# coding: utf-8
import sys

sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
import numpy as np
import operator
from math import log
import treePlotter
import pickle

def getDataAndLabel(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = np.array(data)
    labels = (ndata[:,30:]).T
    result = ndata[:,:30]
    return result,labels[0]

def getData(fileName):
    data = Data.getDataFromFile(fileName)
    return data

def getFeatureName():
    featName = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port',
                'HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow',
                'Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result',]
    return featName


# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
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
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInFoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInFoGain):
            bestInFoGain = infoGain
            bestFeature = i
    return bestFeature

#多数投票决定该节点类型
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建决策树
def createTree(dataSet, featName):
    classList = [example[-1] for example in dataSet]
    #当类别完全相同时，则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有时，若分类还未结束，则返回出现次数最多的作为该节点类型
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatName = featName[bestFeat]
    myTree = {bestFeatName:{}}
    #得到该特征对应的特征值
    del(featName[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subFeat = featName[:]
        myTree[bestFeatName][value] = createTree(
            splitDataSet(dataSet,bestFeat,value),subFeat
        )
    return myTree

#使用决策树来判断一条数据类别
def classify(inputTree, featLabels,testVec):
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

def storeTree(inputTree, fileName):
    fw = open(fileName,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

def trainDecTree(trainFile, storeFileName):
    #训练决策树
    trainData = getData(trainFile)
    trainFeatName = getFeatureName()
    trainTree = createTree(trainData, trainFeatName)
    storeTree(trainTree,storeFileName)
    treePlotter.createPlot(trainTree)

def testDecTree(testFile, treeFile):
    #测试决策树的正确率
    testData, testLabel = getDataAndLabel(testFile)
    featName = getFeatureName()
    decTree = grabTree(treeFile)
    dataNum = len(testData)
    wrongNum = 0
    for i in range(dataNum):
        testResult = classify(decTree, featName, testData[i])
        trueResult = testLabel[i]
        if int(testResult) != int(trueResult):
            wrongNum+=1
            print '第 %d 条测试数据的预测结果为：%d, 实际结果为: %d' %(i,testResult,trueResult)
    print '共测试%d条数据，错误%d条数据，错误率为%f' %(dataNum, wrongNum, wrongNum/float(dataNum))


def mainFunc():

    #测试决策树准确率
    testDecTree('test_500.txt','train_500_tree.txt')

    return 0

mainFunc()

