# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:43:08 2019

@author: ZhangChunli
"""
import ClusterWay.KANN_DbScan as kd
import numpy as np
from threading import Thread
import time

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
        
def loadDataSetComplication(fileName, splitChar='\t'):
    """
    description: 从文件读入数据集
    :param fileName: 文件名
    :param splitChar: 字符串分隔符
    :return: 数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(curline)
            formatData(fltline)
            dataSet.append(fltline)
    return dataSet

def formatData(rowList):
    baseList = ['DAY','EVENING','MIDNIGHT','ARSON','ASSAULT W/DANGEROUS WEAPON','BURGLARY','HOMICIDE','MOTOR VEHICLE THEFT','ROBBERY','SEX ABUSE','THEFT F/AUTO','THEFT/OTHER']
    #索引0~2为时间段：白天，晚上，半夜
    #索引3~11为犯罪类型：纵火，使用危险武器攻击，入室盗窃，蓄意杀人，机动车盗窃，抢劫，性虐待，汽车盗窃，其他盗窃
    rowListLength = len(rowList)
    for i in range(rowListLength):
        if baseList.__contains__(rowList[i]):
            rowList[i] = baseList.index(rowList[i])  #+1,此处是否加1作用并不大，为方便建立字典就不加了
        else:
            rowList[i] = round(float(rowList[i]), 6)#经纬度保留到小数点之后6位便可精确到1m，够用了
def dividedData(dataSet):
    DaySection = []
    CrimeType = []
    Coordinate = []
    Latitude = []
    for item in dataSet:
        DaySection.append(item[0])
        CrimeType.append(item[1])
        Coordinate.append(item[2:4])
        Latitude.append(item[4:])
    return DaySection,CrimeType,Coordinate,Latitude

def getTargetPointOneDimension(pointData, targetList, targetValue):
    matPointData = np.mat(pointData).transpose()
    matTargetList = np.mat(targetList).transpose()
    #positionArray = np.nonzero(matTargetList == targetValue)[0]
    resultSet = matPointData[:, np.nonzero(matTargetList == targetValue)[0]]#现在是一个两行n列的矩阵,第一行经度，第二行纬度
    #print(resultSet)
    #print(resultSet.transpose())
    #print(resultSet.transpose().tolist())
    return resultSet.transpose().tolist()

def getOneDimensionDict(pointData, DaySection, CrimeType):
    classfyDict = {}
    baseList = ['DAY','EVENING','MIDNIGHT','ARSON','ASSAULT W/DANGEROUS WEAPON','BURGLARY','HOMICIDE','MOTOR VEHICLE THEFT','ROBBERY','SEX ABUSE','THEFT F/AUTO','THEFT/OTHER']
    for i in range(0,3):
        classfyDict[baseList[i]] = getTargetPointOneDimension(pointData, DaySection, i)
    for i in range(3,len(baseList)):
        classfyDict[baseList[i]] = getTargetPointOneDimension(pointData, CrimeType, i)
    return classfyDict
def main():
    threads = []#建立线程池
    thread_result = {"Cluster": [],"Eps": [],"MinPts":[]}
    dataSet = loadDataSetComplication('filter_2k/DataSet_all.csv', splitChar=',')
    DaySection, CrimeType, Coordinate, Latitude = dividedData(dataSet)#把数据按照类型分开，实际为分列
    DictByClass = getOneDimensionDict(Latitude, DaySection, CrimeType)#按照类别筛选数据，不同类别分别存放，得到一个字典
    #多线程运行不同维度数据以提高运算效率
    
    baseList = ['DAY','EVENING','MIDNIGHT','ARSON','ASSAULT W/DANGEROUS WEAPON','BURGLARY','HOMICIDE','MOTOR VEHICLE THEFT','ROBBERY','SEX ABUSE','THEFT F/AUTO','THEFT/OTHER']
    for item in baseList:
        tread_tmp = MyThread(kd.kannDbscan, args=(DictByClass[item], item))
        tread_tmp.start()
        threads.append(tread_tmp)#建立线程开启线程之后,都扔进线程池
    
    for i in range(len(baseList)):
        threads[i].join()   #主线程等待
        Cluster_tmp,Eps_tmp,Minpts_tmp = threads[i].get_result()
        thread_result['Cluster'].append(Cluster_tmp)
        thread_result['Eps'].append(Eps_tmp)
        thread_result['MinPts'].append(Eps_tmp)
    #print(DictByClass)
    #kd.showInitPoint(Coordinate)
    #kd.showInitPoint(Latitude)
    #DayData = getTargetPointOneDimension(Latitude, DaySection, 0)
    #Cluster,Eps,Minpts = kd.kannDbscan(Latitude, "Latitude")
    print(thread_result)
    #print(dataSet)

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print('finish all in %s' % str(end - start))