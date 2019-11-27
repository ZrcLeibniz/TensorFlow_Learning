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
from sklearn.preprocessing import StandardScaler

for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[: 5000], x_train_all[5000:]
y_valid, y_train = y_train_all[: 5000], y_train_all[5000:]
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)
print(x_train.shape, y_train.shape)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

# 模型的构建

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 定义完模型后就可以开始进行训练
history = model.fit(x_train_scaled, y_train, epochs=3, validation_data=(x_valid_scaled, y_valid))


def plot_learning_curves(History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

print(model.evaluate(x_test_scaled, y_test))
