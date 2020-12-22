#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/12/21 8:54
# @Author  :xiaoqianke

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as py
from random import randrange
import matplotlib.pyplot as plt


def generateData():
    """测试数据"""

    def get(start, end):
        return [randrange(start, end) for _ in range(50)]

    x1 = get(0, 40)
    x2 = get(70, 100)
    y1 = get(0, 30)
    y2 = get(40, 70)

    data2 = list(zip(x1, y1)) + list(zip(x1, y2)) + list(zip(x2, y1)) + list(zip(x2, y2))
    return py.array(data2)


def dbscan(data1, eps=0.3, min_samples=10):
    # 聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data1)
    # 标记核心对象对应下表为true
    coreSamplesMask = py.zeros_like(db.labels_, dtype=bool)
    coreSamplesMask[db.core_sample_indices_] = True
    # 聚类标签（数组，表示每个样本所属聚类）和所有聚类的数量，标签-1对应的样本表示噪点
    clusterLabels = db.labels_
    uniqueClusterLabels = set(clusterLabels)
    nClusters = len(uniqueClusterLabels) - (-1 in clusterLabels)

    # 绘制聚类结果
    colors = ['red', 'green', 'blue', 'black', 'gray', '#ff00ff', '#ffff00']
    markers = ['v', '^', 'o', '*', 'x', 'h', 'd']
    for i, cluster in enumerate(uniqueClusterLabels):
        print('聚类标签为{}的数据'.format(cluster).center(40, '='))
        #
        clusterIndex = (clusterLabels == cluster)
        #
        coreSamples = data1[clusterIndex & coreSamplesMask]
        # print('核心对象'.1just(30, '*'))
        print(coreSamples)
        plt.scatter(coreSamples[:, 0], coreSamples[:, 1], c=colors[i], marker=markers[i], s=80)

        noiseSamples = data1[clusterIndex & ~coreSamplesMask]
        print(noiseSamples)
        plt.scatter(noiseSamples[:, 0], noiseSamples[:, 1], c=colors[i], marker=markers[i], s=26)
    plt.show()


data = generateData()
dbscan(data, 10, 15)
