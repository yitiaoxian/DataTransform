#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2021/1/5 18:44
# @Author  :xiaoqianke
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载diabetes数据集
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 仅适用一个特征
diabetes_X = diabetes_X[:, np.newaxis, 2]

# 将数据分为训练集/测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标(targets)分为训练集/测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 创建线性回归对象
regr = linear_model.LinearRegression()

# 使用训练集训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用测试集进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)

# 系数
print('Coefficients: \n', regr.coef_)
# 均方误差
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 确定系数：1为完美预测
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# 绘制输出结果
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()