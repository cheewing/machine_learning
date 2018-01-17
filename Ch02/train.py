# -*- coding:UTF-8 -*-

import numpy as np
import operator

def classify0(inX, datingDataMat, datingLabelVec, k):
	m = datingDataMat.shape[0]
	# 求待测点和所有点的距离
	diffMat = np.tile(inX, (m, 1)) - datingDataMat
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	# 距离排序--升序,下标数组
	sortedDistDicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteILabel = datingLabelVec[sortedDistDicies[i]]
		classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
	# iteritems()可迭代元素
	# operator.itemgetter(1) 第2列元素比较
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def createDataSet():
	dataMat = np.array([[1.0, 1.0], [1.0, 1.1], [0.0, 1.0], [0.0, 1.1]])
	labelVec = ['love', 'love', 'hate', 'hate']
	return dataMat, labelVec

def file2matrix(filename):
	fr = open(filename)
	lines = fr.readlines()
	m = len(lines)
	dataMat = np.zeros((m, 3))
	labelVec = []
	index = 0
	for line in lines:
		listFromLine = line.strip().split('\t')
		dataMat[index, :] = listFromLine[0:3]
		labelVec.append(int(listFromLine[-1]))
		index += 1
	return dataMat, labelVec

def autoNorm(dataSet):
	m = dataSet.shape[0]
	minVec = dataSet.min(0)	# 按列取最小值数组
	maxVec = dataSet.max(0) # 按列取最大值数组
	ranges = maxVec - minVec
	normData = (dataSet - np.tile(minVec, (m, 1))) / np.tile(ranges, (m, 1))
	return normData, minVec, ranges

def datingClassTest():
	hoRatio = 0.5
	datingDataMat, datingLabelVec = file2matrix('datingTestSet2.txt')
	normDataMat, minVec, ranges = autoNorm(datingDataMat)
	m = normDataMat.shape[0]
	mTest = int(m * hoRatio)
	errorCount = 0
	for i in range(mTest):
		classifierResult = classify0(normDataMat[i, :], normDataMat[mTest:m, :], datingLabelVec[mTest:m], 3)
		print "the classifier result is %d, and the real class is %d" % (classifierResult, datingLabelVec[i])
		if classifierResult != datingLabelVec[i]:
			errorCount += 1
	print "the total errorCount is %d" % errorCount
	print "the total error rate is %f" % (errorCount / float(mTest))

def datingClassTestSklearn():
	from sklearn.neighbors import KNeighborsClassifier as kNN
	hoRatio = 0.50
	datingDataMat, datingLabelVec = file2matrix('datingTestSet2.txt')
	normDataMat, minVec, ranges = autoNorm(datingDataMat)
	m = normDataMat.shape[0]
	mTest = int(m * hoRatio)
	errorCount = 0
	#构建kNN分类器
	neigh = kNN(n_neighbors = 3, algorithm = 'auto')
	#拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
	neigh.fit(normDataMat, datingLabelVec)
	print normDataMat
	for i in range(mTest):
		classifierResult = neigh.predict([normDataMat[i, :].tolist()])
		print "the classifier result is %d, and the real class is %d" % (classifierResult, datingLabelVec[i])
		if classifierResult != datingLabelVec[i]:
			errorCount += 1
	print "the total errorCount is %d" % errorCount
	print "the total error rate is %f" % (errorCount / float(mTest))

def img2vector(filename):
	returnVec = np.zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		line = fr.readline()
		for j in range(32):
			returnVec[0, i*32+j] = int(line[j])
	return returnVec

def handwritingClassTest():
	from os import listdir
	from sklearn.neighbors import KNeighborsClassifier as kNN
	# 获取训练数据
	# 获取训练样本文件集合
	trainFileList = listdir('trainingDigits')
	dataSize = len(trainFileList)
	trainMat = np.zeros((dataSize, 1024))
	trainLabelVec = []
	for i in range(dataSize):
		filename = trainFileList[i]
		classNumber = int(filename.split('_')[0])
		trainLabelVec.append(classNumber)
		trainMat[i,:] = img2vector('trainingDigits/%s' % filename)
	neigh = kNN(n_neighbors = 3, algorithm = 'auto')
	neigh.fit(trainMat, trainLabelVec)
	# 获取测试样本文件集合
	testFileList = listdir('testDigits')
	mTest = len(testFileList)
	errorCount = 0
	for i in range(mTest):
		filename = testFileList[i]
		classNumber = int(filename.split('_')[0])
		testVec = img2vector('testDigits/%s' % filename)
		#classifierResult = classify0(testVec, trainMat, trainLabelVec, 3)
		classifierResult = neigh.predict(testVec)
		print "the classifier result is %d, and the real class is %d" % (classifierResult, classNumber)
		if classifierResult != classNumber:
			errorCount += 1.0
	print "the total error count is: %d" % errorCount
	print "the total error rate is: %f" % (errorCount / float(mTest))

def testfile2matrix():
	filename = 'datingTestSet2.txt'
	dataMat, labelVec = file2matrix(filename)
	print dataMat
	print labelVec


if __name__ == '__main__':
	handwritingClassTest()
