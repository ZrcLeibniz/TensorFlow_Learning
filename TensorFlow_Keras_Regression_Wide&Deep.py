# 实现wide&deep模型

import tensorflow as ts
import sklearn as sk
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入加利福尼亚州的房价

housing_price = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing_price.data, housing_price.target, random_state=7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)
normal = StandardScaler()
x_train_normal = normal.fit_transform(x_train)
x_valid_normal = normal.transform(x_valid)
x_test_normal = normal.transform(x_test)

# 由于本模型是由两部分组成，所以不能使用Sequential方式来实现
# 使用函数式API/功能API来构建wide-deep模型
input_data = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_data)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_data, hidden2])
output_data = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input_data], outputs=[output_data])
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train_normal, y_train, epochs=10, validation_data=(x_valid_normal, y_valid))


def show_history(History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


show_history(history)

