# -*- coding:UTF-8 -*-
from math import log

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['age', 'have job', 'have house', 'loan']        #特征标签
    return dataSet, labels                #返回数据集和分类属性


def calcShannonEnt(dataSet):
	m = len(dataSet)
	featLabelVec = [example[-1] for example in dataSet]
	classCount = {}
	for i in range(m):
		label = featLabelVec[i]
		classCount[label] = classCount.get(label, 0) + 1
	shannonEnt = 0.0
	for key in classCount:
		prob = classCount[key] / float(m)
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		classCount[vote] = classCount.get(vote, 0) + 1
	sortedClassCount = sorted(classList.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def createTree(dataSet, labels, featLabels):
	classList = [example[-1] for example in dataSet]
	# 停止条件
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
	return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def classifySklearn():
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	from sklearn.externals.six import StringIO
	from sklearn import tree
	import pandas as pd
	import numpy as np
	import pydotplus
	with open('lenses.txt', 'r') as fr:                                        #加载文件
		lenses = [inst.strip().split('\t') for inst in fr.readlines()]        #处理文件
	lenses_target = []                                                        #提取每组数据的类别，保存在列表里
	for each in lenses:
		lenses_target.append(each[-1])

	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']            #特征标签       
	lenses_list = []                                                        #保存lenses数据的临时列表
	lenses_dict = {}                                                        #保存lenses数据的字典，用于生成pandas
	for each_label in lensesLabels:                                            #提取信息，生成字典
		for each in lenses:
			lenses_list.append(each[lensesLabels.index(each_label)])
		lenses_dict[each_label] = lenses_list
		lenses_list = []
    # print(lenses_dict)                                                        #打印字典信息
	lenses_pd = pd.DataFrame(lenses_dict)                                    #生成pandas.DataFrame
	print(lenses_pd)                                                        #打印pandas.DataFrame
	le = LabelEncoder()                                                        #创建LabelEncoder()对象，用于序列化            
	for col in lenses_pd.columns:                                            #为每一列序列化
		lenses_pd[col] = le.fit_transform(lenses_pd[col])
	print(lenses_pd)

	clf = tree.DecisionTreeClassifier(max_depth = 4)                        #创建DecisionTreeClassifier()类
	clf = clf.fit(lenses_pd.values.tolist(), lenses_target)                    #使用数据，构建决策树
	dot_data = StringIO()
	tree.export_graphviz(clf, out_file = dot_data,                            #绘制决策树
							feature_names = lenses_pd.keys(),
							class_names = clf.classes_,
							filled=True, rounded=True,
							special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf("tree.pdf")    

if __name__ == "__main__":
	dataSet, labels = createDataSet()
	print calcShannonEnt(dataSet)
	print splitDataSet(dataSet, 0, 1)
	print chooseBestFeatureToSplit(dataSet)
	featLabels = []
	myTree = createTree(dataSet, labels, featLabels)
	print(myTree)

	classifySklearn()
