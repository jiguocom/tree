from math import log
import operator

"""
决策树类
"""
class DecisionTree:

    def __init__(self):
        pass

    # 创建决策树
    def createTree(self, dataSet, labels):

        classList = [example[-1] for example in dataSet]  # 类别：男或女

        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)  # 选择最优特征

        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}  # 分类结果以字典形式保存
        del (labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet \
                                                          (dataSet, bestFeat, value), subLabels)

        return myTree


    # 计算数据的熵(entropy)
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)  # 数据条数
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]  # 每行数据的最后一个字（类别）
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1  # 统计有多少个类以及每个类的数量
        shannonEnt = 0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries  # 计算单个类的熵值
            shannonEnt -= prob * log(prob, 2)  # 累加每个类的熵值
        return shannonEnt

    # 按某个特征分类后的数据
    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    # 选择最优的分类特征
    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)  # 原始的熵

        bestInfoGain = 0
        bestFeature = -1

        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)

                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)  # 按特征分类后的熵

            infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
            if (infoGain > bestInfoGain):  # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    '''用决策树分类函数'''
    def classify(self, inputTree, featLabels, testVec):

        if type(inputTree) == str:
            return ""

        firstStr = list(inputTree.keys())[0]  # 获取tree的根节点对于的key值

        secondDict = inputTree[firstStr]  # 通过key得到根节点对应的value

        # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
        featLabels = [firstStr] if len(featLabels) == 0 else featLabels

        featIndex = featLabels.index(firstStr)

        classLabel = ""
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]

        return classLabel
