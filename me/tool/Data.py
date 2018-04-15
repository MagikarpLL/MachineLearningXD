#coding: utf-8
import random


def selectSomeDataFromFile(fileName, numberOfData):
    path = '/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/data/old/'
    currPath = path + fileName
    with open(currPath,'r') as f:
        data = f.readlines()
        result = random.sample(data,numberOfData)
    return result

def selectAndDeleteSomeDataFromFile(fileName, numberOfData):
    path = '/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/data/old/'
    currPath = path + fileName
    with open(currPath,'r') as f:
        data = f.readlines()
    random.shuffle(data)
    result = data[:numberOfData]
    data = data[numberOfData:]
    with open(currPath,'w') as file:
        for line in data:
            file.write(line)
    return result


def saveDataToFile(fileName,data):
    path = '/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/data/current/'
    currPath = path + fileName
    with open(currPath,'w') as f:
        for lines in data:
            f.write(lines)


def getDataFromFile(fileName):
    path = '/home/magikarpll/me/workplace/pycharm/MachineLearningXD/me/data/current/'
    currPath = path + fileName
    result = []
    with open(currPath,'r') as f:
       for line in f:
           nline = line.strip("\r\n")
           temp_list = [int(x) for x in nline.split(',')]
           #print temp_list
           result.append(temp_list)
    return result