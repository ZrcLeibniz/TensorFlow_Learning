import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
from tensorflow import keras
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

for mode in tf, np, pd, sk, keras, mlt:
    print(mode.__name__, mode.__version__)


Fashion_data = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = Fashion_data.load_data()
x_valid, x_train = x_train_all[: 500], x_train_all[500:]
y_valid, y_train = y_train_all[: 500], y_train_all[500:]

normal = StandardScaler()
x_train_normal = normal.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_normal = normal.fit_transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_normal = normal.fit_transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train_normal, y_train, epochs=10, validation_data=(x_valid_normal, y_valid))


