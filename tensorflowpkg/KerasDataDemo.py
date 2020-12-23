#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/11/12 8:37
# @Author  :xiaoqianke

import tensorflow as tf
import keras as ks

mnist = tf.keras.datasets.mnist

(xtrain,ytrain),(x_test,y_test) = mnist.load_data()

xtrain,x_test = xtrain / 255.0 ,x_test / 255.0

#将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数：
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
##训练并验证模型
model.fit(xtrain,ytrain,epochs=5)
model.evaluate(x_test,y_test,verbose=2)
