# coding: utf-8
import sys

sys.path.append(r"/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/tool/")
import Data
from numpy import *
from math import log,sqrt
import random
import pickle

#加载数据
def getData(fileName):
    data = Data.getDataFromFile(fileName)
    return data

#切割数据集
def cross_validation_split(dataSet, n_folds):
    dataSet_split = list()
    dataSet_copy = list(dataSet)
    fold_size = len(dataSet)/n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))
        dataSet_split.append(fold)
    return dataSet_split

#计算准确率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual)) * 100.0

#
def evaluate_algorithm(dataSet, algorithm, n_folds,*args):
    folds = cross_validation_split(dataSet, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set,[])
