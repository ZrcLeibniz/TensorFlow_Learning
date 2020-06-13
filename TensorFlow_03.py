import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist

(train_images_all, train_labels_all), (test_images, test_label) = fashion.load_data()
train_images, valid_images = train_images_all[500:1500], train_images_all[: 500]
train_labels, valid_labels = train_labels_all[500:1500], train_labels_all[: 500]

normal = StandardScaler()
train_images_normal = normal.fit_transform(train_images.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
test_images_normal = normal.fit_transform(test_images.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
valid_images_normal = normal.fit_transform(valid_images.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)

model = keras.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
print(model.summary())
model.compile(optimizer=tf.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(train_images_normal, train_labels, epochs=30, validation_data=(valid_images_normal, valid_labels))
loss = model.evaluate(test_images_normal, test_label)
print(loss)


def plot_learning_curves(History):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)
