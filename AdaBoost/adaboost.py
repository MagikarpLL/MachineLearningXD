# coding: utf-8
from numpy import *


def loadSimpData():
    dataMat = matrix([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0],
                      ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels



#单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,  i, threshVal, inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                #print "split: dim %d, thresh %.2f, thresh inequal: %s, thr weighted error is %.3f" %\
                (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst



#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        alpha = float(0.5 * log(1-error)/max(error,1e-16))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst:", classEst.T
        #为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        #错误率累加计算
        aggClassEst += alpha * classEst
        print "aggClassEst:", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels), ones((m,1)))
        errorsRate = aggErrors.sum()/m
        print "total error:", errorsRate, "\n"
        if errorsRate == 0.0:break
    return weakClassArr