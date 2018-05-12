# Random Forest Algorithm on Sonar Dataset
# coding: utf-8
from me.tool import Data
from random import seed
from random import randrange
from math import sqrt
import pickle
import time


# get data
def getData(fileName):
    data = Data.getDataFromFile(fileName)
    return data


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy[index])
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    trees = algorithm(dataset, *args)
    return trees

    '''
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, trees = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores, trees
    '''


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    gini = 0.0
    D = len(groups[0]) + len(groups[1])
    for class_value in classes:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            propotion = [row[-1] for row in group].count(class_value) / float(size)
            gini += float(size) / D * (propotion * (1 - propotion))
    return gini

    '''
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
    '''


# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
        print '第' + str(i) + '颗完成'

    # predictions = [bagging_predict(trees, row) for row in test]
    # return (predictions), trees
    return trees


def storeTree(inputTree, fileName):
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


def trainForest(trainFile, storeFile, max_depth, n_trees):
    # Test the random forest algorithm
    seed(2)
    # load and prepare data
    dataset = getData(trainFile)
    # evaluate algorithm
    n_folds = 5
    min_size = 1
    sample_size = 0.8
    n_features = int(sqrt(len(dataset[0]) - 1))

    storeFile = str(max_depth) + '/' + storeFile

    time_start = time.time()

    trees = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)

    time_end = time.time()
    print time_end - time_start

    print 'depth_' + str(max_depth) + '_num_' + str(n_trees) + '_complete'
    storeTree(trees, storeFile)


def testForest(treeFile, testFile):
    trees = grabTree(treeFile)
    dataSet = getData(testFile)
    classLabels = [row[-1] for row in dataSet]

    allNum = len(dataSet)
    errorNum = 0
    fnNum = 0
    tpNum = 0

    for i in range(len(dataSet)):
        dataSet[i][-1] = None
    predictions = [bagging_predict(trees, row) for row in dataSet]

    for i in range(allNum):
        if (classLabels[i] != predictions[i]):
            errorNum += 1
            if (classLabels[i] == -1):
                fnNum += 1
        else:
            if (classLabels[i] == -1):
                tpNum += 1

    print treeFile + '||||' + '错误率: %f, 误判率: %f' % (float(errorNum) / allNum, fnNum / float(fnNum + tpNum))



def trainAllFun(depth):

    # trainForest('train_3000_1.txt', 'forest_10_3000_1.txt', depth, 10)
    # trainForest('train_3000_2.txt', 'forest_10_3000_2.txt', depth, 10)
    # trainForest('train_3000_3.txt', 'forest_10_3000_3.txt', depth, 10)
    # trainForest('train_3000_4.txt', 'forest_10_3000_4.txt', depth, 10)
    #
    # trainForest('train_3000_1.txt', 'forest_30_3000_1.txt', depth, 30)
    # trainForest('train_3000_2.txt', 'forest_30_3000_2.txt', depth, 30)
    # trainForest('train_3000_3.txt', 'forest_30_3000_3.txt', depth, 30)
    # trainForest('train_3000_4.txt', 'forest_30_3000_4.txt', depth, 30)
    #
    # trainForest('train_3000_1.txt', 'forest_60_3000_1.txt', depth, 60)
    trainForest('train_3000_2.txt', 'forest_60_3000_2.txt', depth, 60)
    trainForest('train_3000_3.txt', 'forest_60_3000_3.txt', depth, 60)
    trainForest('train_3000_4.txt', 'forest_60_3000_4.txt', depth, 60)

    #trainForest('train_3000_1.txt', 'forest_120_3000_1.txt', depth, 120)
    #trainForest('train_3000_2.txt', 'forest_120_3000_2.txt', depth, 120)
    #trainForest('train_3000_3.txt', 'forest_120_3000_3.txt', depth, 120)
    #trainForest('train_3000_4.txt', 'forest_120_3000_4.txt', depth, 120)
    return

def testAllFuc(depth):
    # testForest(str(depth) + '/forest_10_3000_1.txt', 'test_3000_1.txt')
    # testForest(str(depth) + '/forest_10_3000_2.txt', 'test_3000_2.txt')
    # testForest(str(depth) + '/forest_10_3000_3.txt', 'test_3000_3.txt')
    # testForest(str(depth) + '/forest_10_3000_4.txt', 'test_3000_4.txt')
    #
    # testForest(str(depth) + '/forest_30_3000_1.txt', 'test_3000_1.txt')
    # testForest(str(depth) + '/forest_30_3000_2.txt', 'test_3000_2.txt')
    # testForest(str(depth) + '/forest_30_3000_3.txt', 'test_3000_3.txt')
    # testForest(str(depth) + '/forest_30_3000_4.txt', 'test_3000_4.txt')
    #
    # testForest(str(depth) + '/forest_60_3000_1.txt', 'test_3000_1.txt')
    testForest(str(depth) + '/forest_60_3000_2.txt', 'test_3000_2.txt')
    testForest(str(depth) + '/forest_60_3000_3.txt', 'test_3000_3.txt')
    testForest(str(depth) + '/forest_60_3000_4.txt', 'test_3000_4.txt')

    #testForest(str(depth) + '/forest_120_3000_1.txt', 'test_3000_1.txt')
    #testForest(str(depth) + '/forest_120_3000_2.txt', 'test_3000_2.txt')
    #testForest(str(depth) + '/forest_120_3000_3.txt', 'test_3000_3.txt')
    #testForest(str(depth) + '/forest_120_3000_4.txt', 'test_3000_4.txt')
    return

def RandomForestMain():
    # trainAdaCart('train_500_1.txt', 'test.txts' , 3, 10)

    testAllFuc(5)
    #testAllFuc(8)

    #trainAllFun(6)
    #trainAllFun(8)
    #trainAllFun(5)
    # trainAllFun(10)
    return


def mainFuc(trainFile, storeFile, treeFile, testFile):
    # trainForest(trainFile, storeFile)
    testForest(treeFile, testFile)


# mainFuc('train_500.txt','forest_500.txt', 'forest_500.txt','test_500.txt')
RandomForestMain()
