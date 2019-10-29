# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:43:08 2019

@author: ZhangChunli
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = False
NOISE = 0

def loadDataSet(fileName, splitChar='\t'):
    """
    description: 从文件读入数据集
    :param filename: 文件名
    :param splitChar: 字符串分隔符
    :return: 数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet

def dist(a, b):
    """
    description: 求两对象距离
    :param a: 一个一行的矩阵
    :param b: 一个一行的矩阵
    :return: a,b两对象(两个向量)的欧式距离
    """
    return math.sqrt(np.power(a - b, 2).sum())#此时的a，b均是矩阵

def eps_neighbor(a, b, eps):#判断b与a的密度直达性
    """
    description: 判断b与a的密度直达性
    :param a: 一个一行的矩阵
    :param b: 一个一行的矩阵
    :param eps: 约束直达性的最小半径
    :return: True或False代表a,b 是否直达
    """
    return dist(a, b) < eps 

def region_query(data, pointId, eps):
    """
    description: 查找与piontId对应点密度直达的所有点的id
    :param data: 所有的数据集
    :param pointId: 查询点id
    :param eps: 约束直达性的最小半径
    :return: 在eps范围内的点的id集合
    """
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i) #记录的是i，把i扔到seeds列表里面去
    return seeds

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    """
    description: 从piontId开始进行扩张，形成聚类簇
    :param data: 所有的数据集
    :param clusterResult: 一个列表，表示对应id的对象的身份
    :param pointId: 查询点id
    :param clusterId: 簇id
    :param eps: 约束直达性的最小半径
    :param minPts: 邻域内最少个数
    :return: True或False代表能否扩张与聚类
    """
    seeds = region_query(data, pointId, eps)#找寻ponitId的直达点
    if len(seeds) < minPts: # 不满足minPts条件的为噪声点
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId #ponitId确定为可分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0: # 持续扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint) #把扩张所得到的点扔进seeds里面，这样就可以不用迭代了
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId #噪声点不具备扩展的潜能，直接扔进聚类簇就行了，不用再扔进seeds里面了
            seeds = seeds[1:] #把第一个扩张完成seed给去掉，进行下一次扩张
        return True #直到一个point无法扩张，返回True

def dbscan(data, eps, minPts):
    """
    description: 从piontId开始进行扩张，形成聚类簇
    :param data: 所有的数据集
    :param eps: 约束直达性的最小半径
    :param minPts: 邻域内最少个数
    :return: 两个结果，第一个为聚类结果，id位置存放自己所属的簇，第二个为聚类簇的数量
    """
    clusterId = 1 #第n个簇
    nPoints = data.shape[1]#读取data矩阵第1维的长度，应该是数据集个数
    clusterResult = [UNCLASSIFIED] * nPoints 
    for pointId in range(nPoints):#pointId代表第n个对象,代表一个对象
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1

def plotFeature(data, clusters, clusterNum):
    """
    description: 把聚类结果画出来
    :param clusters: 聚类结果列表，id位置存放自己所属的簇
    :param clusterNum: 聚类簇的数量
    :return: 
    """
    #nPoints = data.shape[1] #获取对象个数
    matClusters = np.mat(clusters).transpose()  #对聚类结果cluster进行转置，应该变成一个列向量了
    fig = plt.figure() #建立一个 matplotlib.figure.Figure对象，是所有绘图元素的顶层容器
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111) #在图中添加一个Axes对象作为子图布置的一部分，Axes对象包含大部分figure Elements
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        #print("np.nonzero(matClusters[:, 0].A == i):",np.nonzero(matClusters[:, 0].A == i))
        subCluster = data[:, np.nonzero(matClusters == i)] #参考的原写法是后面这样的，但是发现好像并没有太多作用subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        #matClusters[:, 0].A是将matrix类型转换成array类型,但这个时候A仍然是二维的虽然仅有一行，A[0]就变成了一行的一维array类型
        #matClusters == i 会返回一个n行一列的矩阵，n等于i的位置为True其他位置为false
        # np.nonzero() 输入数组或矩阵，以矩阵的形式返回输入值中非零元素的信息索引，返回两个一维矩阵，第一维指定输入矩阵的非零元素所在行，第二维指定输入矩阵非零元素所在列
        #data[:, ] 
        #  ****原始写法***  
        #ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=5)
        #print("\n横坐标：",subCluster[0, :].flatten().A[0])
        #print("\n纵坐标：",subCluster[1, :].flatten().A[0])
        #   *******更新写法**********
        x = subCluster[0, :].flatten().A[0]
        y = subCluster[1, :].flatten().A[0]
        x_suit_len = int(np.shape(x)[0]/2)
        y_suit_len = int(np.shape(y)[0]/2)
        ax.scatter(x[0:x_suit_len], y[0:y_suit_len], c=colorSytle, s=15)
        #Axes.scatter(x,y,c=None,s=None), 功能为画散点图x,y分别为对应横纵坐标列表，c为颜色或颜色列表，s为画的点的大小，可以为一常数或列表
        #subCluster[0, :].flatten()把矩阵变成降维到一维，但此时还是一个矩阵
        #subCluster[0, :].flatten().A[0] 把降维到一维的矩阵变成列表
        #这一部分原始写法存在一个问题，第一个点会多画非常多遍，有多少个对象重画多少遍，不知道为什么会多产生一维数据,因为上面就多一维，取一个[0]就好了其实
        #print("\n横坐标：",x[0:x_suit_len])
        #print("\n纵坐标：",y[0:y_suit_len])
def main():
    dataSet = loadDataSet('betterTestData/788points.txt', splitChar=',')
    #print(dataSet)
    dataSet = np.mat(dataSet).transpose()#把列表变成矩阵，并把矩阵转置，形成每行为一个对象的两个属性值，新矩阵为两行n列
    #print(dataSet)
    #clusters, clusterNum = dbscan(dataSet, 2.851, 33)
    #clusters, clusterNum = dbscan(dataSet, 0.01077557864522233, 33.26143790849673)
    clusters, clusterNum = dbscan(dataSet, 2.24, 24)
    print("cluster Numbers = ", clusterNum)
    #print("cluster Result = ", clusters)
    
    plotFeature(dataSet, clusters, clusterNum) #将聚类结果传给制图程序

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print('finish all in %s' % str(end - start))
    plt.show()


