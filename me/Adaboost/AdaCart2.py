# coding: utf-8
import sys

sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
import pickle
from itertools import *
import operator, time, math
import matplotlib.pyplot as plt
from numpy import *


def getDataAndLabel(fileName):
    data = Data.getDataFromFile(fileName)
    ndata = array(data)
    labels = (ndata[:, 30:]).T
    result = ndata[:, :30]
    return result, labels[0]

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    """stumpClassify(将数据集，按照feature列的value进行 二分法切分比较来赋值分类)
    Args:
        dataMat    Matrix数据集
        dimen      特征列
        threshVal  特征列要比较的值
    Returns:
        retArray 结果集
    """
    # 默认都是1
    retArray = ones((shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, labelArr, D):
    """buildStump(得到决策树的模型)
    Args:
        dataArr   特征标签集合
        labelArr  分类标签集合
        D         最初的样本的所有特征权重集合
    Returns:
        bestStump    最优的分类器模型
        minError     错误率
        bestClasEst  训练后的结果集
    """
    # 转换数据
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).T
    # m行 n列
    m, n = shape(dataMat)

    # 初始化数据
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    # 初始化的最小误差为无穷大
    minError = inf

    # 循环所有的feature列，将列切分成 若干份，每一段以最左边的点作为分类节点
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        # print 'rangeMin=%s, rangeMax=%s' % (rangeMin, rangeMax)
        # 计算每一份的元素个数
        stepSize = (rangeMax-rangeMin)/numSteps
        # 例如： 4=(10-1)/2   那么  1-4(-1次)   1(0次)  1+1*4(1次)   1+2*4(2次)
        # 所以： 循环 -1/0/1/2
        for j in range(-1, int(numSteps)+1):
            # go over less than and greater than
            for inequal in ['lt', 'gt']:
                # 如果是-1，那么得到rangeMin-stepSize; 如果是numSteps，那么得到rangeMax
                threshVal = (rangeMin + float(j) * stepSize)
                # 对单层决策树进行简单分类，得到预测的分类值
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                # print predictedVals
                errArr = mat(ones((m, 1)))
                # 正确为0，错误为1
                errArr[predictedVals == labelMat] = 0
                # 计算 平均每个特征的概率0.2*错误概率的总和为多少，就知道错误率多高
                # 例如： 一个都没错，那么错误率= 0.2*0=0 ， 5个都错，那么错误率= 0.2*5=1， 只错3个，那么错误率= 0.2*3=0.6
                weightedError = D.T*errArr
                '''
                                dim            表示 feature列
                                threshVal      表示树的分界值
                                inequal        表示计算树左右颠倒的错误率的情况
                                weightedError  表示整体结果的错误率
                                bestClasEst    预测的最优结果
                                '''
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

                # bestStump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少
            return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, labelArr, numIt=40):
    """adaBoostTrainDS(adaBoost训练过程放大)
    Args:
        dataArr   特征标签集合
        labelArr  分类标签集合
        numIt     实例数
    Returns:
        weakClassArr  弱分类器的集合
        aggClassEst   预测的分类结果值
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化 D，设置每行数据的样本的所有特征权重集合，平均分为m份
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 得到决策树的模型
        bestStump, error, classEst = buildStump(dataArr, labelArr, D)

        # alpha 目的主要是计算每一个分类器实例的权重(加和就是分类结果)
        # 计算每个分类器的 alpha 权重值
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        # store Stump Params in Array
        weakClassArr.append(bestStump)

        # print "alpha=%s, classEst=%s, bestStump=%s, error=%s " % (alpha, classEst.T, bestStump, error)
        # 分类正确：乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
        # 分类错误：乘积为 -1，结果会受影响，所以也乘以 -1
        expon = multiply(-1 * alpha * mat(labelArr).T, classEst)
        # print '\n'
        # print 'labelArr=', labelArr
        # print 'classEst=', classEst.T
        # print '\n'
        # print '乘积: ', multiply(mat(labelArr).T, classEst).T
        # 判断正确的，就乘以-1，否则就乘以1， 为什么？ 书上的公式。
        # print '(-1取反)预测值expon=', expon.T
        # 计算e的expon次方，然后计算得到一个综合的概率的值
        # 结果发现： 判断错误的样本，D对于的样本权重值会变大。
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # print "D: ", D.T
        # print '\n'

        # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
        # print '当前的分类结果：', alpha*classEst.T
        aggClassEst += alpha * classEst
        # print "叠加后的分类结果aggClassEst: ", aggClassEst.T
        # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
        # 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
        aggErrors = multiply(sign(aggClassEst) != mat(labelArr).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print "total error=%s " % (errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
    # do stuff similar to last aggClassEst in adaBoostTrainDS
    dataMat = mat(datToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    # 循环 多个分类器
    for i in range(len(classifierArr)):
        # 前提： 我们已经知道了最佳的分类器的实例
        # 通过分类器来核算每一次的分类结果，然后通过alpha*每一次的结果 得到最后的权重加和的值。
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                     classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print aggClassEst
    return sign(aggClassEst)





def storeTree(inputTree, fileName):
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


def trainAdaCart(trainData, storeName, depth, num):
    dataSet, labelSet = getDataAndLabel(trainData)
    weakClassArr, aggClassEst = adaBoostTrainDS(dataSet, labelSet, num)
    print weakClassArr, '\n-----\n', aggClassEst.T
    storeTree(weakClassArr, storeName)
    return


def testAdaCart(testData, treeFile):
    tree = grabTree(treeFile)
    dataSet, labelSet = getDataAndLabel(testData)
    predicting10 = adaClassify(dataSet, tree)
    m = len(dataSet)
    errArr = mat(ones((m, 1)))
    # 测试：计算总样本数，错误样本数，错误率
    print m, errArr[predicting10 != mat(labelSet).T].sum(), errArr[predicting10 != mat(labelSet).T].sum() / m



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
    #trainAdaCart('train_500_1.txt', 'test.txt' , 3, 300)
    error = testAdaCart('train_1000_1.txt', 'test.txt')

    #trainAllFun(3)
    #trainAllFun(5)
    #trainAllFun(8)
    #trainAllFun(10)
    return


AdaCartMain()
