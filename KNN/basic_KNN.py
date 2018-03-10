#coding:utf-8

from matplotlib import pyplot
from numpy import *
import operator

def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


#inX为用于分类的输入向量
#dataSet为输入的训练样本集
#lables为标签向量
#参数k表示用于选择最近邻居的数目
def classify0(inX, dataSet, lables, k):
    #array的shape函数返回指定维度的大小，如dataset为n*m的矩阵，
    #则dataset.shape[0]返回n,dataset.shape[1]返回m,dataset.shape返回n,m
    dataSetSize = dataSet.shape[0]
    # tile函数简单的理解，它的功能是重复某个数组。比如tile(A,n)，功能是将数组A重复n次，构成一个新的数组
    # 所以此处tile(inX,(dataSetSize,1))的作用是将inX重复复制dataSetSize次，以便与训练样本集的样本个数一致

    # 减去dataSet就是求出其差值，所以diffMat为一个差值矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # 以下三行代码执行的是欧式距离的计算
    sqDiffMat = diffMat ** 2
    # 平时用的sum应该是默认的axis=0,就是普通的相加,而当加入axis=1以后就是将一个矩阵的每一行向量相加,axis用于控制是行相加还是列相加
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    #相关性的排序
    # argsort????x????????????????????????????????????
    # x=np.array([1,4,3,-1,6,9])
    # x.argsort() ?? ([3,0,2,1,4,5])
    sortedDistance = distance.argsort()
    # 确定前K个点所在类别出现的频率
    classCount = {}
    # range(K) ,如果k=3, 那么i=0,1,2
    for i in range(k):
        voteLable = lables[sortedDistance[i]]
        # dict.get(key, default=None)key 为字典中要查找的键，default如果指定键的值不存在时，返回该默认值值。此句代码用于统计标签出现的次数
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
    # sorted函数参数解释，sorted(iterable, cmp=None, key=None, reverse=False)
    # iterable：是可迭代类型;
    # cmp：用于比较的函数，比较什么由key决定;
    # key：用列表元素的某个属性或函数进行作为关键字，有默认值，迭代集合中的一项;
    # reverse：排序规则. reverse = True  降序 或者 reverse = False 升序，有默认值。
    # 返回值：是一个经过排序的可迭代类型，与iterable一样。
    ######
    # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号）
    ######
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回最符合的标签
    return sortedClassCount[0][0]


group, lables = createDataset()
# 画出点的分布
pyplot.plot(group[:, 0], group[:, 1], 'ro', label='point')
pyplot.ylim(-0.2, 1.2)
pyplot.xlim(-0.2, 1.2)
pyplot.show()

# 测试[0,0]所属类别
print classify0([0, 0], group, lables, 3)
