import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from tensorflow import keras
import pprint
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

for module in tf, np, pd, mpl, sk, keras:
    print(module.__name__, module.__version__)

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

pprint.pprint(housing.data[0: 5])
pprint.pprint(housing.target[0: 5])

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_valid.shape)

normal = StandardScaler()
x_train_normal = normal.fit_transform(x_train)
x_valid_normal = normal.transform(x_valid)
x_test_normal = normal.transform(x_test)

model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]))
model.add(keras.layers.Dense(1))
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]
history = model.fit(x_train_normal, y_train, validation_data=(x_valid, y_valid), callbacks=callbacks, epochs=10)


def plot_learning_curves(History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

history2 = model.evaluate(x_test_normal, y_test)

