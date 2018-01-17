# -*- coding:UTF-8 -*-
import numpy as np
import operator

def createDataSet():
    dataMat = np.array([[1.0,1.1], [1.0,1.0], [0.0,1.0], [0.0,1.1]])
    labelVec = ['love', 'love', 'hate', 'hate']
    return dataMat, labelVec



def classify0(inX, dataMat, labelVec, k):
    # 训练集的个数
    dataSize = dataMat.shape[0]
    # 扩展inX到训练集的维度
    inXMat = np.tile(inX, (dataSize, 1))
    # inX跟训练集所有点计算距离
    diffMat = inXMat - dataMat
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 按照距离排序，取前k个最近的点
    for i in range(k):
        voteILabel = labelVec[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    # 统计k个最近点的类别数，最多的类别即为inX的类别
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
归一化
'''
def autoNorm(dataMat):
    # 训练集大小
    dataSize = dataMat.shape[0]
    # 每列的最小值
    minVec = np.min(dataMat, axis = 0)
    minMat = np.tile(minVec, (dataSize, 1))
    # 每列的最大值
    maxVec = np.max(dataMat, axis = 0)
    maxMat = np.tile(maxVec, (dataSize, 1))
    # 最大值-最小值
    diffMat = maxMat - minMat
    # 训练集-最小值
    diffDataMat = dataMat - minMat
    # 归一化
    normMat = diffDataMat / diffMat

    return normMat

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    dataMat = np.zeros((numberOfLines, 3))
    classLabelVec = []
    index = 0
    for line in lines:
        listFromLine = line.strip().split('\t')
        dataMat[index, :] = listFromLine[0:3]
        classLabelVec.append(int(listFromLine[-1]))
        index += 1
    return dataMat, classLabelVec

def datingClassTest():
    hoRatio = 0.50
    datingDataMat, datingLabelVec = file2matrix('datingTestSet2.txt')
    # 归一化
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 抽样测试样本
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabelVec[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabelVec[i])
        if (classifierResult != datingLabelVec[i]): 
            errorCount += 1.0
    print "the total error rate is:%f" % (errorCount/float(numTestVecs))
    print errorCount

def img2vector(filename):
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, i * 32 + j] = int(lineStr[i])
    return returnVec

def handwritingClassTest():
    from os import listdir
    from sklearn.neighbors import KNeighborsClassifier as kNN
    hwLabels = []
    # 目录下文件名
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    # 获取训练数据和标签
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    #构建kNN分类器
    neigh = kNN(n_neighbors = 10, algorithm = 'auto')
    #拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest);
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))
    return trainingFileList

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
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    #设置汉字格式
    #font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
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
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
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

def showDataPlot():
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.ylabel('some numbers')
    plt.show()

def showDifPlot():
    import numpy as np
    import matplotlib.pyplot as plt

    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()

def showPic():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    img=mpimg.imread('stinkbug.png')
    print img
    imgplot = plt.imshow(img)
    plt.show()

    # RGB单色图
    lum_img = img[:, :, 0]
    plt.imshow(lum_img)
    plt.show()

    # 热力图
    plt.imshow(lum_img, cmap="hot")
    plt.show()

    # 
    imgplot = plt.imshow(lum_img)
    imgplot.set_cmap('nipy_spectral')
    plt.colorbar()
    plt.show()

def showHistogram():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    img=mpimg.imread('stinkbug.png')
    lum_img = img[:, :, 0]
    plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()

def showComplex():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    img=mpimg.imread('stinkbug.png')
    lum_img = img[:, :, 0]
    imgplot = plt.imshow(lum_img, clim=(0.0, 0.7))
    plt.show()

def showImages():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    from PIL import Image
    img = Image.open('stinkbug.png')
    img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-place
    imgplot = plt.imshow(img, interpolation="bicubic")
    plt.show()

def showSubplot():
    import matplotlib.pyplot as plt
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    plt.show()

def showGredSpec():
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :-1])
    ax3 = plt.subplot(gs[1:, -1])
    ax4 = plt.subplot(gs[-1, 0])
    ax5 = plt.subplot(gs[-1, -2])
    plt.show()

def showDemoGridSpec():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    try:
        from itertools import product
    except ImportError:
        # product is new in v 2.6
        def product(*args, **kwds):
            pools = map(tuple, args) * kwds.get('repeat', 1)
            result = [[]]
            for pool in pools:
                result = [x+[y] for x in result for y in pool]
            for prod in result:
                yield tuple(prod)


    def squiggle_xy(a, b, c, d, i=np.arange(0.0, 2*np.pi, 0.05)):
        return np.sin(i*a)*np.cos(i*b), np.sin(i*c)*np.cos(i*d)

    fig = plt.figure(figsize=(8, 8))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(4, 4, wspace=0.0, hspace=0.0)

    for i in range(16):
        inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3,
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)
        a, b = int(i/4)+1,i%4+1
        for j, (c, d) in enumerate(product(range(1, 4), repeat=2)):
            ax = plt.Subplot(fig, inner_grid[j])
            ax.plot(*squiggle_xy(a, b, c, d))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    all_axes = fig.get_axes()

    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    plt.show()



if __name__ == '__main__':
    handwritingClassTest()