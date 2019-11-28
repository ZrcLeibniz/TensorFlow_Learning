# encoding=utf-8
# 测试回调函数

import tensorflow as tf
import pandas as pd
import matplotlib as mlt
import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import os

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[: 500], x_train_all[500:]
y_valid, y_train = y_train_all[: 500], y_train_all[500:]
normal = StandardScaler()
x_train_normal = normal.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_normal = normal.fit_transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_normal = normal.fit_transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)



model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimize='adam', metrics=['accuracy'])

# TensorBoard EarlyStopping ModelCheckPoint
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'fashion_mnist_model.h5')

callback = [keras.callbacks.TensorBoard(logdir),
            keras.callbacks.ModelCheckpoint('output_model_file', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, min_delta=1e-3)]
model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), callbacks=callback)
