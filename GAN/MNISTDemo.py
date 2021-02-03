#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2021/1/25 9:41
# @Author  :yitiaoxian


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame, Series
from keras import models, layers, optimizers, losses, metrics
from keras.utils.np_utils import to_categorical

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# GAN生成器网络
# generator 模型，它将一个向量（来自潜在空间，训练过程中对其随机 采样）转换为一张候选图像。GAN常见的诸多问题之一，就是生成器“卡在”看似噪声的生成图像上。一种可行的解决方案是在判别器和生成器中都使用 dropout。
import keras

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)  # 将输入转换为大小为 16×16 的 128 个通道的特征图

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)  # 上采样为32*32
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)  # 生成一个大小为 32×32 的单通道特征图 （即 CIFAR10 图像的形状）
generator = keras.models.Model(generator_input, x)
generator.summary()

# GAN判别器网络
# discriminator模型，它接收一张候选图像（真实的或合成的）作为输入，并将其划分到这两个类别之一：“生成图像”或“来自训练集的真实图像”

discrimination_input = layers.Input(shape=(height, width, channels))  # 判别器输入为生成图像与真实图像的拼接，以判断图像的‘真假’
x = layers.Conv2D(128, 3)(discrimination_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)  # 卷积窗口4*4，步幅为2
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)  # 分类层（真或假）
discriminator = keras.models.Model(discrimination_input, x)  # 将判别器模型实例化，这里它将形状为 (32, 32, 3)的输入转换为一个二进制分类决策（真/假）
discriminator.summary()
discriminator_optimizer = optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,  # 优化器中使用梯度裁剪（限制梯度的范围）[它是一个动态的系统，其最优化过程寻找的不是一个最小值，而是两股力量之间的平衡。]
    decay=1e-8  # 为了稳定训练过程，使用学习率衰减
)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# 对抗网络
'''
    最后，我们要设置GAN，将生成器和判别器连接在一起。
    训练时，这个模型将让生成器向某个方向移动，从而提高它欺骗判别器的能力。这个模型将潜在空间的点转换为一个分类决策（即“真”或“假”），它训练的标签都是“真实图像”。
    因此，训练 gan 将会更新 generator 的权重，使得 discriminator 在观察假图像时更有可能预测为“真”。
    请注意，有一点很重要，就是在训练过程中需要将判别器设置为冻结（即不可训练），这样在训练 gan 时它的权重才不会更新。
    如果在此过程中可以对判别器的权重进行更新，那么我们就是在训练判别器始终预测“真”，但这并不是我们想要的！
'''
discriminator.trainable = False  # 将判别器权重设置为不可训练 （仅应用于 gan 模型）

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(
    lr=0.0004,
    clipvalue=1.0,
    decay=1e-8
)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# 训练DCGAN
'''
每轮都进行以下操作：
    (1) 从潜在空间中抽取随机的点（随机噪声）。
    (2) 利用这个随机噪声用 generator 生成图像。
    (3) 将生成图像与真实图像混合。
    (4) 使用这些混合后的图像以及相应的标签（真实图像为“真”，生成图像为“假”）来训练 discriminator，如图 8-18 所示。
    (5) 在潜在空间中随机抽取新的点。
    (6) 使用这些随机向量以及全部是“真实图像”的标签来训练gan。这会更新生成器的权重（只更新生成器的权重，因为判别器在 gan中被冻结），其更新方向是使得判别器能够将生成图像预测为“真实图像”。这个过程是训练生成器去欺骗判别器。
'''
import os
import keras
from keras.preprocessing import image

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]  # 选择青蛙图像（类别编号为 6）
print(x_train.shape)  # (5000, 32, 32, 3)
x_train = x_train.reshape(
    (x_train.shape[0],) +
    (height, width, channels)).astype('float32') / 255.  # 数据标准化
iterations = 10000
batch_size = 20
save_dir = 'datasets/gan_output'
start = 0  # 记录当前批处理的位置
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))  # 潜在空间中采样随机点
    generated_images = generator.predict(random_latent_vectors)  # 利用生成器解码为虚假图像
    stop = start + batch_size
    real_images = x_train[start:stop]  #
    combined_images = np.concatenate([generated_images, real_images])  # 拼接，默认0轴（纵向）
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])  # 列向量，1表示生成的图像，0表示真实的图像
    labels += 0.05 * np.random.random(labels.shape)  # 向标签中添加随机噪声
    d_loss = discriminator.train_on_batch(combined_images, labels)  # 返回判别器损失：使用的是二进制交叉熵
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
    a_loss = gan.train_on_batch(  # 通过gan模型训练生成器
        random_latent_vectors,
        misleading_targets  # 冻结判别器权重（置0）
    )
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:  # 每100步保存并绘图
        gan.save_weights('gan.h5')  # 保存模型权重
        print('discriminator loss:', d_loss)
        print('adversarial loss', a_loss)
        img = image.array_to_img(generated_images[0] * 255., scale=False)  # 转换成图像并保存
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))
