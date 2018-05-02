# coding: utf-8
import sys

sys.path.append(r"I:\Workplace\WorkPlace\PyCharm\MachineLearningXD\me\tool")
import Data
from numpy import *
import operator
from math import log
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
def calcShannonEnt(dataSet, classLabels, subD):
    numEntries = len(dataSet)
    labelCounts = set(classLabels)
    shannonEnt = 0.0
    for item in labelCounts:
        numItem = ones(numEntries)
        for i in range(numEntries):
            if classLabels[i] != item:
                numItem[i] = 0
        weights = mat(numItem)
        weights = weights * subD
        weight = (weights.getA())[0][0]
        numTemp = numItem.tolist().count(1)
        prob = numTemp/float(numEntries)
        if(prob != 0.0):
            shannonEnt -= prob * weight *log(prob, 2)
    return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet, classLabels , axis, value, D):
    retDataSet = []
    tempD = ((D.T).getA())[0]
    resultD = []
    resultLabels = []
    for featVec in dataSet:
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
    return bestFeature

def createTree(trainData, classLabels ,featName, depth ,D):

    #print 'createTree'

    #当类别完全相同时，停止继续划分
    tempSet = set(classLabels)
    if(len(tempSet) == 1):
        return classLabels[0]
    #遍历完所有时，若分类还未结束，则返回出现次数最多的作为该节点类型
    if len(trainData) == 1:
        return majorityCnt(classLabels)
    if depth == 0:
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
            subDataSet ,subClassLabels,subFeat, depth - 1 ,subD
        )
    return myTree

def classify(inputTree, featLabels, testVec):

    keyList = inputTree.keys()
    if len(keyList)>= 2:
        keyList.remove('alpha')
    firstStr = keyList[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    classLabel = 1

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def buildDecTree(trainData, classLabels, featName, depth , D):
    featNameLabels = featName[:]
    decTree = createTree(trainData, classLabels ,featNameLabels, depth ,D)
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


def adaBoostTrainDT(trainData, classLabels, featName,depth, numIt=40):
    weakClassArr = []
    m = shape(trainData)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        decTree, error, classEst = buildDecTree(trainData, classLabels, featName, depth ,D)
        print "D:", D.T
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        decTree['alpha'] = alpha
        weakClassArr.append(decTree)
        print 'depth: ' + str(depth) + '—num:' + str(numIt) + '第' + str(i) +'个完成'
        print "classEst:", classEst
        # 为下一次迭代计算D
        # 此处的multiply为对应元素相乘
        expon = multiply(-1 * alpha * mat(classLabels), classEst)
        # exp(x)返回x的指数
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 错误率累加计算
        aggClassEst += alpha * mat(classEst).T
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        #if errorRate <= 0.13: break
    return weakClassArr

def getDataAndLabel(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = array(data)
    labels = (ndata[:,30:]).T
    result = ndata[:,:30]
    return result.tolist(),labels[0].tolist()

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

def trainAda(dataFile,storeFile, depth, num):
    trainData, trainLabel = getDataAndLabel(dataFile)
    featName = getFeatureName()

    storeFile = 'depth_' + str(depth) + '/' + storeFile

    weakClassArr = adaBoostTrainDT(trainData, trainLabel, featName, depth ,num)
    storeTree(weakClassArr, storeFile)
    return 0

def classifySelf(inputTree, featLabels, testVec):
    value = 0.0
    for i in range(len(inputTree)):
            value = value + inputTree[i].get('alpha') * classify(inputTree[i], featLabels, testVec)
    if value >= 0:
        return 1
    else:
        return -1

def testAda(treeFile, testFile, depth):
    treeFile = 'depth_' + str(depth) + '/' + treeFile
    tree = grabTree(treeFile)
    testData, testLabel = getDataAndLabel(testFile)
    featName = getFeatureName()

    errorNum = 0
    allNum = len(testData)
    fnNum = 0
    tpNum = 0

    for i in range(len(testData)):
        classRet = classifySelf(tree, featName, testData[i])
        if classRet != testLabel[i]:
            errorNum += 1
            if(testLabel[i] == -1):
                fnNum += 1
        else:
            if(testLabel[i] == -1):
                tpNum += 1
    print treeFile + '|||||' + '错误率为: %f, 误判率为: %f' %(float(errorNum)/allNum, fnNum/float(fnNum + tpNum))

def trainAllFuc(depth):
    trainAda('train_3000_1.txt','ada_3000_10_1.txt', depth, 10)
    trainAda('train_3000_2.txt','ada_3000_10_2.txt', depth, 10)
    trainAda('train_3000_3.txt','ada_3000_10_3.txt', depth, 10)
    trainAda('train_3000_4.txt','ada_3000_10_4.txt', depth, 10)

    trainAda('train_3000_1.txt','ada_3000_30_1.txt', depth, 30)
    trainAda('train_3000_2.txt','ada_3000_30_2.txt', depth, 30)
    trainAda('train_3000_3.txt','ada_3000_30_3.txt', depth, 30)
    trainAda('train_3000_4.txt','ada_3000_30_4.txt', depth, 30)

    trainAda('train_3000_1.txt','ada_3000_60_1.txt', depth, 60)
    trainAda('train_3000_2.txt','ada_3000_60_2.txt', depth, 60)
    trainAda('train_3000_3.txt','ada_3000_60_3.txt', depth, 60)
    trainAda('train_3000_4.txt','ada_3000_60_4.txt', depth, 60)

    #trainAda('train_3000_1.txt','ada_3000_120_1.txt', depth, 120)
    #trainAda('train_3000_2.txt','ada_3000_120_2.txt', depth, 120)
    #trainAda('train_3000_3.txt','ada_3000_120_3.txt', depth, 120)
    #trainAda('train_3000_4.txt','ada_3000_120_4.txt', depth, 120)

    return

def testAllFuc(depth):
    testAda('ada_3000_10_1.txt','test_3000_1.txt', depth)
    #testAda('ada_3000_10_2.txt','test_3000_2.txt', depth)
    #testAda('ada_3000_10_3.txt','test_3000_3.txt', depth)
    #testAda('ada_3000_10_4.txt','test_3000_4.txt', depth)

    testAda('ada_3000_30_1.txt','test_3000_1.txt', depth)
   # testAda('ada_3000_30_2.txt','test_3000_2.txt', depth)
    #testAda('ada_3000_30_3.txt','test_3000_3.txt', depth)
    #testAda('ada_3000_30_4.txt','test_3000_4.txt', depth)

    testAda('ada_3000_60_1.txt','test_3000_1.txt', depth)
    #testAda('ada_3000_60_2.txt','test_3000_2.txt', depth)
   # testAda('ada_3000_60_3.txt','test_3000_3.txt', depth)
   # testAda('ada_3000_60_4.txt','test_3000_4.txt', depth)

    #trainAda('train_3000_1.txt','ada_3000_120_1.txt', depth, 120)
    #trainAda('train_3000_2.txt','ada_3000_120_2.txt', depth, 120)
    #trainAda('train_3000_3.txt','ada_3000_120_3.txt', depth, 120)
    #trainAda('train_3000_4.txt','ada_3000_120_4.txt', depth, 120)

    return

def mainFunc():
    #训练分类器
    trainAda('train_500_1.txt','test.txt', 3, 10)
    #trainAllFuc(3)

    #trainAllFuc(5)

    #testAllFuc(3)

    #测试决策树准确率
    #testAda('ada_test_1000.txt','test_2000.txt')

    return 0

mainFunc()
