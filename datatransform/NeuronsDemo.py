#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/11/11 8:58
# @Author  :xiaoqianke

import numpy as np

def sigmod(x):
    #激活函数f(x)=1/(1+e^-x)
    return 1/(1+np.exp(-x))

class Neuron():
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias

    def feedback(self,inputs):
        total = np.dot(self.weights,inputs)+self.bias
        return sigmod(total)

weights = np.array([0.5,0.5])#w1=0,w2=1
bias = 4
n = Neuron(weights,bias)
#inputs
x = np.array([2,3])
print(n.feedback(x))
