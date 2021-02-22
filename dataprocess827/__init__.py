#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/12/23 8:23
# @Author  :xiaoqianke

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

df = pd.DataFrame(pd.read_csv('F://58_2//labels_train.csv'))

#print(df.shape)
print(df.info())
print(df.dtypes)
#print(df.iloc[:2])

#训练集 测试集 验证集的划分
train,test = train_test_split(df,test_size=0.3)
train,val = train_test_split(train,test_size=0.3)

#shuffle 洗牌
#原始数据的处理
def df_to_dataset(dataframe,shuffle=True,batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('ALARM_STATUS')
    labels1 = labels.map({'报警':1,'无报警':0})
    #print(labels)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels1))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5 # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#for feature_batch, label_batch in train_ds.take(1):
#  print('Every feature:', list(feature_batch.keys()))
#  print('A batch of MACH_XCOR:', feature_batch['MACH_XCOR'])
#  print('A batch of targets:', label_batch )

feature_columns = []
# 数值列
for header in ['MACH_XCOR', 'MACH_YCOR', 'MACH_ZCOR', 'ABS_XCOR', 'ABS_YCOR', 'ABS_ZCOR', 'RELV_XCOR','RELV_YCOR',
               'RELV_ZCOR','DIST_XCOR','DIST_YCOR','DIST_ZCOR']:
  feature_columns.append(feature_column.numeric_column(header))
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential(
    [
        feature_layer,
        layers.Dense(128,activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=False)
model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)