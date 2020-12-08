#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/11/5 14:09
# @Author  :xiaoqianke

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf

train = pd.read_csv("F:/58_2/labels_train.csv",encoding='GB2312')

X_train,X_test = train_test_split(train,test_size=0.32,random_state=42)
print(X_train)
print(X_test)
