# 使用tensorFlow自定义损失函数
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as mlt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
normal_tool = StandardScaler()
x_train_normal = normal_tool.fit_transform(x_train)
x_test_normal = normal_tool.fit_transform(x_test)
x_valid_normal = normal_tool.fit_transform(x_valid)


def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y_true))

model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation='selu', input_shape=x_test_normal.shape[1:]))
model.add(keras.layers.Dense(1))
model.compile(loss=customized_mse, optimizer='sgd', metrics=['mean_squared_error'])
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
history = model.fit(x_train_normal, y_train, epochs=10, validation_data=(x_valid_normal, y_valid),
                    callbacks=callbacks)
