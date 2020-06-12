from keras_preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf


train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
path1 = 'E:/traindata/train'
path2 = 'E:/traindata/valida'
# dst_path = 'E:/traindataPlus'

train_generator = train_datagen.flow_from_directory(
    directory=path1,
    target_size=(300, 300),
    class_mode='binary',
    batch_size=1,
    # save_to_dir=dst_path
)

valid_generator = valid_datagen.flow_from_directory(
    directory=path2,
    target_size=(300, 300),
    class_mode='binary',
    batch_size=1,
)

model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
print(model.summary())

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=3,
    epochs=15,
    validation_data=valid_generator,
    validation_steps=1,
    verbose=2
)
