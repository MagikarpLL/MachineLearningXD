# coding: utf-8
import sys

sys.path.append(r"I:\Workplace\WorkPlace\PyCharm\MachineLearningXD\me\tool")
import Data
from numpy import *
import operator
from math import log
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

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
def calcShannonEnt(dataSet, classLabels, D):
    numEntries = len(dataSet)
    labelCounts = set(classLabels)
    shannonEnt = 0.0
    for item in labelCounts:
        errArr = ones(numEntries)
        for i in range(numEntries):
            if classLabels[i] != item:
                errArr[i] = 0
        errArr = mat(errArr)
        prob = errArr * D
        prob = (prob.getA())[0][0]
        if(prob != 0.0):
            shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet, classLabels , axis, value, D):
    retDataSet = []
    tempD = ((D.T).getA())[0]
    resultD = []
    resultLabels = []
    for featVec in dataSet:

        #print 'for'

        if(featVec[axis] == value):
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
            resultD.append(tempD[dataSet.index(featVec)])
            resultLabels.append(classLabels[dataSet.index(featVec)])
    return retDataSet,mat(resultD).T, resultLabels

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet,classLabels,D):
    numFeatures = len(dataSet[0])



    baseEntropy = calcShannonEnt(dataSet, classLabels, D)



    bestInfoGain = 0.0
    bestFeature = -1

    startT = tm.time()

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet, subD, subClassLabels = splitDataSet(dataSet,classLabels, i, value, D)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet, subClassLabels, subD)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    endT = tm.time()
    print "for Time:",(endT - startT)

    return bestFeature

def createTree(trainData, classLabels ,featName, D):

    print 'createTree'

    #当类别完全相同时，停止继续划分
    tempSet = set(classLabels)
    if(len(tempSet) == 1):
        return classLabels[0]
    #遍历完所有时，若分类还未结束，则返回出现次数最多的作为该节点类型
    if len(trainData) == 1:
        return majorityCnt(classLabels)

    #startT = tm.time()

    bestFeat = chooseBestFeatureToSplit(trainData,classLabels, D)

    #endT = tm.time()
    #print 'chooseBestFeatureToSplit Time is:',(endT-startT)

    bestFeatName = featName[bestFeat]
    myTree = {bestFeatName:{}}
    del(featName[bestFeat])
    featValues = [example[bestFeat] for example in trainData]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subFeat = featName[:]

        #startT2 = tm.time()

        subDataSet, subD, subClassLabels = splitDataSet(trainData,classLabels, bestFeat,value, D)

        #endT2 = tm.time()
        #print 'splitDataSet Time is:', (endT2 - startT2)


        myTree[bestFeatName][value] = createTree(
            subDataSet ,subClassLabels,subFeat, subD
        )
    return myTree

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
    featNameLabels = featName[:]
    decTree = createTree(trainData, classLabels ,featNameLabels, D)
    print 'train one dec tree successfully'
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
        print "classEst:", classEst
        # 为下一次迭代计算D
        # 此处的multiply为对应元素相乘
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        # exp(x)返回x的指数
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 错误率累加计算
        aggClassEst += alpha * mat(classEst).T
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        if errorRate <= 0.13: break
    return weakClassArr

def getDataAndLabel(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = array(data)
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

def storeTree(inputTree, fileName):
    fw = open(fileName,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

def adaBoostApi(X, y):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=200, learning_rate=0.8)
    bdt.fit(X, y)
    return bdt

def trainAda(dataFile,testFile):
    trainData, trainLabel = getDataAndLabel(dataFile)
    testData, testLabel = getDataAndLabel(testFile)

    bdt = adaBoostApi(trainData, trainLabel)

    result =  bdt.predict(testData)

    errorNum = 0
    for i in range(len(testLabel)):
        if testLabel[i] != result[i]:
            errorNum += 1

    print errorNum/float(len(testLabel))

    #weakClassArr = adaBoostTrainDT(testData, testLabel, featName,20)
    #storeTree(weakClassArr, storeFileName)
    return 0

def mainFunc():
    #训练分类器
    trainAda('test_500.txt','test_2000.txt')
    #测试决策树准确率
    #testDecTree('test_500.txt','train_500_tree.txt')

    return 0

mainFunc()
