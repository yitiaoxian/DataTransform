#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/12/23 8:23
# @Author  :xiaoqianke

import pandas as pd
import numpy as np

def dataClean():
    csv_data = pd.read_csv('../data2/labels_train.csv','gbk')
    colsName = csv_data.columns
    print(colsName)
    print(csv_data)

def dataSingle():
    #处理单行数据
    print('单行处理')

def dataMulti():
    #处理多行数据
    print('多行处理')

dataClean()