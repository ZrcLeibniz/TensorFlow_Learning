# encoding=utf-8
# 使用Keras搭建分类模型

import matplotlib as mpl
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow import keras


for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[: 5000], x_train_all[5000:]
y_valid, y_train = y_train_all[: 5000], y_train_all[5000:]
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)
print(x_train.shape, y_train.shape)


def show_single_image(ima_arr):
    # 可以使用这个方法来传入一个numpy类型的序列，来显示图片
    plt.imshow(ima_arr, cmap="binary")
    plt.show()


def show_images(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * n_rows + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()


Class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show_single_image(x_train[0])
# show_images(3, 5, x_train[0: 10], y_train[0: 10], Class_names)

# 模型的构建
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.layers)
# 可以显示出来模型的架构
print(model.summary())
# 定义完模型后就可以开始进行训练
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_valid, y_valid))

print(history.history)


def plot_learning_curves(History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

