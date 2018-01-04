#-*- coding: UTF-8 -*-
from numpy import *
import operator

"""
函数说明:创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
Modify:
    2018-01-04
"""
def createDataSet():
    dataset = array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['love', 'love', 'action', 'action']
    return dataset, labels

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果

Modify:
    2017-07-13
"""
def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetRows = dataSet.shape[0]

    # 在列向量方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次（纵向）
    inputDataSet = tile(inX, [dataSetRows, 1]) 

    # 计算距离
    # 二维特征相减后，求平方和
    diffMat = dataSet - inputDataSet
    squareDiffMat = diffMat ** 2
    squareDistances = squareDiffMat.sum(axis = 1)
    distances = squareDistances ** 0.5

    # 根据距离排序，返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()

    # 定义一个类别次数的字典
    classCount = {}

    # 取前k个距离最近的数据，找出类别次数最多的类
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # key = operator.itemgetter(1) 根据字典的值排序
    # key = operator.itemgetter(0) 根据字典的键排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

    # 返回次数最多的分类
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #打印数据集
    print group
    print labels
    test = [101, 20]

    # kNN分类
    test_class = classify0(test, group, labels, 3)

    # 打印分类结果
    print test_class