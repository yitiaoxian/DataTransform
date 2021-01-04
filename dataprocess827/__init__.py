#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/12/23 8:23
# @Author  :xiaoqianke

import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv('F://58_2//labels_train2.csv'))

#print(df.shape)
#print(df.info())
#print(df.dtypes)
print(df.iloc[:2])