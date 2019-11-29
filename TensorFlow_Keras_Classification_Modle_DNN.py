import tensorflow as tf
import numpy as np
import matplotlib as mlt
import pandas as pd
from tensorflow import keras
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


Fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = Fashion_mnist.load_data()
x_train, x_valid = x_train_all[500:], x_train_all[: 500]
y_train, y_valid = x_train_all[500:], x_train_all[: 500]

normal = StandardScaler()
x_train_normal = normal.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_normal = normal.fit_transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_normal = normal.fit_transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

for mode in tf, np, mlt, pd, keras, sk:
    print(mode.__name__, mode.__version__)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimize='adam', metrics=['accuracy'])

logdir = 'log'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'fashion_mnist_model_h5')

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_valid_normal, y_valid), callbacks=callbacks)
