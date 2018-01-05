#-*- coding: UTF-8 -*-
from numpy import *
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

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

"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量

Modify:
    2017-03-24
"""
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    lenOfLines = len(arrayOfLines)
    dataMat = zeros((lenOfLines, 3)); # 取全0数组，避免lineData字符串转换为int
    classLabelVector = [] # label向量
    index = 0
    for line in arrayOfLines:
        # strip()去除\r\n，split('\t')按\t划分字符串，得到一个list
        listFromLine = line.strip().split('\t')
        dataMat[index,:] = listFromLine[0:3];
        #vectorOfLabels.append(int(lineData[-1]))
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1      # 此处不能忘

    return array(dataMat), classLabelVector

"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2017-03-24
"""
def autoNorm(dataSet):
    # 每列取最大值，得最大值向量
    maxColVector = dataSet.max(0)
    # 每列取最小值，得最小值向量
    minColVector = dataSet.min(0)
    # 最大值最小值的范围
    ranges = maxColVector - minColVector
    # shape(dataSet)返回dataSet的行列数
    normDataSet = zeros(shape(dataSet))
    # dataSet行数
    m = dataSet.shape[0]
    # 减去最小值
    normDataSet = dataSet - tile(minColVector, (m, 1))
    # normDataSet／ranges矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet

"""
函数说明:可视化数据

Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类Label
Returns:
    无
Modify:
    2017-03-24
"""
def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    #font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) # windows
    font = FontProperties(fname="/Library/Fonts/Songti.ttc", size=14)     # mac
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

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

    data, label = file2matrix('datingTestSet2.txt')
    print data, label

    #打开的文件名
    #filename = "datingTestSet.txt"
    #打开并处理数据
    #datingDataMat, datingLabels = file2matrix(filename)
    #showdatas(datingDataMat, datingLabels)

    nornDataSet = autoNorm(data)
    print nornDataSet