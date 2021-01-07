#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2021/1/7 18:31
# @Author  :yitiaoxian
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# 导入一些数据进行训练
iris = datasets.load_iris()
X = iris.data[:, :2]  # 我们仅使用前两个特征。
Y = iris.target

logreg = LogisticRegression(C=1e5)

# 创建逻辑回归分类器的实例并拟合数据。
logreg.fit(X, Y)

# 绘制决策边界。 为网格[x_min，x_max] x [y_min，y_max]中的每个点分配颜色。
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # 网格中的步长
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# 将结果放入颜色图(color plot)
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# 把训练点也绘制到图中
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()