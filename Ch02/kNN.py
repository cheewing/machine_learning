#-*- coding: UTF-8 -*-
from numpy import *
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
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
        classLabelVector.append(int(listFromLine[-1]))
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        '''
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        '''
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
    maxVals = dataSet.max(0)
    # 每列取最小值，得最小值向量
    minVals = dataSet.min(0)
    # 最大值最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的行列数
    normDataSet = zeros(shape(dataSet))
    # dataSet行数
    m = dataSet.shape[0]
    # 减去最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # normDataSet／ranges矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

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

"""
函数说明:分类器测试函数

Parameters:
    无
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2017-03-24
"""
def datingClassTest():
    # 打开的文件名
    filename = "datingTestSet2.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 取所有数据的10%
    hRatio = 0.10
    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 约会数据总数
    m = normMat.shape[0]
    print datingDataMat.shape
    print m
    # 测试数据总数
    numTestVecs = int(m * hRatio)
    print "test" ,
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("错误率:%f%%" %(errorCount / float(numTestVecs) * 100))

"""
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
    无
Returns:
    无

Modify:
    2017-03-24
"""
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet2.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = array([precentTats, ffMiles, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

Modify:
    2017-07-15
"""
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVector中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1*1024向量
    return returnVect

"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

Modify:
    2017-07-15
"""
def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回traingDigits目录下的文件名
    trainningFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainningFileList)
    # 初始化训练的Mat矩阵，测试集
    trainingMat = zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainningFileList[i]
        # 获得分类的数字
        classNumStr = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumStr)
        # 将每一个文件的1*1024数据存储到trainningMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    #构建kNN分类器
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    #拟合模型，trainningMat为测试矩阵，hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumStr = int(fileNameStr.split('_')[0])
        #获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        #classfierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classfierResult = neigh.predict(vectorUnderTest)
        #print "the claßßssfier came back with: %d, the real answer is: %d" % (classfierResult, classNumStr)
        if (classfierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 打印数据集
    print group
    print labels
    test = [101, 20]

    # kNN分类
    #test_class = classify0(test, group, labels, 3)

    # 打印分类结果
    #print test_class

    #data, label = file2matrix('datingTestSet2.txt')
    #print data, label

    #打开的文件名
    #filename = "datingTestSet.txt"
    #打开并处理数据
    #datingDataMat, datingLabels = file2matrix(filename)
    #showdatas(datingDataMat, datingLabels)

    #nornDataSet, ranges, minVals = autoNorm(data)
    #print nornDataSet
    #print ranges
    #print minVals

    #classifyPerson()

    #vect = img2vector('testDigits/0_13.txt')
    #print vect[0, 32:63]

    handwritingClassTest()