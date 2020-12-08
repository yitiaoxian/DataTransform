#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/11/12 9:31
# @Author  :xiaoqianke

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

#print(dataframe)

#划分训练集。测试集。验证集
train,test = train_test_split(dataframe,test_size=0.3)
train,val = train_test_split(train,test_size=0.3)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe,shuffle=True,batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5
train_ds = df_to_dataset(train,batch_size=batch_size)
val_ds = df_to_dataset(val,shuffle=False,batch_size=batch_size)
test_ds = df_to_dataset(test,shuffle=False,batch_size=batch_size)

for feature_batch , label_batch in train_ds.take(1):
    print('every feature:',list(feature_batch.keys()))
    print('a batch of ages:',feature_batch['age'])
    print('a batch of targets:',label_batch)