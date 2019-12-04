# 在神经网络中添加Dropout

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import sklearn as sk
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

for mode in tf, np, pd, mlt, sk:
    print(mode.__name__, mode.__version__)

Fashion_minst = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = Fashion_minst.load_data()
x_valid, x_train = x_train_all[: 500], x_train_all[500:]
y_valid, y_train = y_train_all[: 500], y_train_all[500:]

normal = StandardScaler()
x_train_normal = normal.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_normal = normal.fit_transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_normal = normal.fit_transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(300, activation='selu'))
model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train_normal, y_train, epochs=3, validation_data=(x_valid, y_valid))

print(model.summary())

def show_history(History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


show_history(history)