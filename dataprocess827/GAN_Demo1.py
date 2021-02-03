#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2021/2/3 15:40
# @Author  :yitiaoxian

import numpy as np
import tensorflow as tf

"""
    生成基于size的，统一分布的数据（size，1）
"""
class GenDataLoader():
    def __init__(self,size = 200,low = -1,high =1):
        self.size = size
        self.low = low
        self.high = high

    def next_batch(self):
        z = np.random.uniform(self.low,self.high,[self.size,1])
        return z

"""
    生成基于mu,sigma,size的正态分布数据(size,1)
"""
class RealDataLoader():
    def __init__(self,size = 200,mu = -1,sigma = 1):
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def next_batch(self):
        data = np.random.normal(self.mu, self.sigma, [self.size, 1])  # (batch_size, size)
        data.sort()
        return data


# MLP - used for D_pre, D1, D2, G networks
def mlp(input,output_dim,is_sigmoid = False):
#在本地范围内构造可学习的参数
    w1 = tf.get_variable("w0",[input.get_shape()[1],6],initializer=tf.random_normal_initializer)
    b1 = tf.get_variable("b0",[6],initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [5,output_dim], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    if is_sigmoid == False:
        fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
    else:
        fc3=tf.nn.sigmoid(tf.matmul(fc2,w3)+b3)
    return fc3, [w1,b1,w2,b2,w3,b3]

def momentum_optimizer(loss,var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch,  # Current index into the dataset.
        epoch // 4,          # Decay step - this decays 4 times throughout training process.
        0.95,                # Decay rate.
        staircase=True)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(loss,global_step=batch,var_list=var_list)
    return optimizer

def weight_variable(shape, name):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial, name=name)
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

def bias_variable(shape, name):
    # initial = tf.constant(0.0, shape=shape)
    # return tf.Variable(initial, name=name)
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

class Generator():
    def __init__(self, inputs, input_size = 1, hidden_size = 6, output_size = 1):
        with tf.variable_scope("generator"):
            weight1 = weight_variable(shape=[input_size, hidden_size], name="weight1") #(size, 100)
            bias1 = bias_variable(shape=[1, hidden_size], name="bias1") #(1, 100)
            weight2 = weight_variable(shape=[hidden_size, hidden_size], name="weight2")
            bias2 = bias_variable(shape=[1, hidden_size], name="bias2")
            weight3 = weight_variable(shape=[hidden_size, output_size], name="weight3")
            bias3 = bias_variable(shape=[1, output_size], name="bias3")
            frac1 = tf.nn.tanh(tf.matmul(inputs, weight1) + bias1, name="frac1")   #(batch_size, 100)
            frac2 = tf.nn.tanh(tf.matmul(frac1, weight2) + bias2, name="frac2")
            frac3 = tf.nn.tanh(tf.matmul(frac2, weight3) + bias3, name="frac3")
            self.frac = frac3
            self.var_list = [weight1, bias1, weight2, bias2, weight3, bias3]
            # self.frac, self.var_list = mlp(inputs, 1)
            self.frac = tf.multiply(self.frac, 5)
    def get_param(self):
        return self.frac, self.var_list

class Discriminator():
    def __init__(self, inputs, input_size = 1, hidden_size = 6):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            weight1 = weight_variable(shape=[input_size, hidden_size], name="weight1") #(size, 100)
            bias1 = bias_variable(shape=[1, hidden_size], name="bias1") #(1, 100)
            weight2 = weight_variable(shape=[hidden_size, hidden_size], name="weight2")
            bias2 = bias_variable(shape=[1, hidden_size], name="bias2")
            weight3 = weight_variable(shape=[hidden_size, 1], name="weight3")
            bias3 = bias_variable(shape=[1, 1], name="bias3")
            frac1 = tf.nn.tanh(tf.matmul(inputs, weight1) + bias1, name="frac1")  # (batch_size, 100)
            frac2 = tf.nn.tanh(tf.matmul(frac1, weight2) + bias2, name="frac2") #range()
            frac3 = tf.nn.sigmoid(tf.matmul(frac2, weight3) + bias3, name="frac3") #range()
            self.frac = frac3
            self.var_list = [weight1, bias1, weight2, bias2, weight3, bias3]
            # self.frac, self.var_list = mlp(inputs, 1, is_sigmoid = True)

    def get_param(self):
        return self.frac, self.var_list

if __name__ == '__main__':
    size = 200
    epoch = 1000    #训练次数
    shape = (size, 1)
    x_node = tf.placeholder(tf.float32, shape=shape)  # input M normally distributed floats
    z_node = tf.placeholder(tf.float32, shape=shape)
    generator = Generator(z_node)
    G, theta_g = generator.get_param()
    discriminator2 = Discriminator(G)
    discriminator1 = Discriminator(x_node)
    D1, theta_d = discriminator1.get_param()
    D2, theta_d = discriminator2.get_param()
    loss_d = tf.reduce_mean(tf.log(D1) + tf.log(1 - D2))
    loss_g = tf.reduce_mean(tf.log(D2))

    # set up optimizer for G,D
    train_op_d = momentum_optimizer(1 - loss_d, theta_d)
    # train_op_d = tf.train.AdamOptimizer(0.001).minimize(loss =1 - loss_d)
    train_op_g = momentum_optimizer(1 - loss_g, theta_g)  # maximize log(D(G(z)))
    # train_op_g = tf.train.AdamOptimizer(0.001).minimize(loss=1 - loss_g)  # maximize log(D(G(z)))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    gen_data_load = GenDataLoader(size)
    real_data_load = RealDataLoader(size)
    for i in range(epoch):
        for j in range(2):
            real_data = real_data_load.next_batch()
            gen_data = gen_data_load.next_batch()
            sess.run([train_op_d, loss_d], {x_node: real_data, z_node: gen_data})
        gen_data = gen_data_load.next_batch()
        sess.run([train_op_g, loss_g], {z_node: gen_data})  # update generator
        if (i % 50 == 0):
            real_data = real_data_load.next_batch()
            D1_, D2_ = sess.run([D1, D2], {x_node: real_data, z_node: gen_data})
            print("epoch:%d " % i, "D1:", D1_[0], ",D2:", D2_[0])
    writer = tf.summary.FileWriter("./graphs/implementation_1_graph", sess.graph)
    writer.close()
    sess.close()
