import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layer

#GLOBAL VARIABLES
path_t = '/home/ubuntu/Training_data'
path_v = '/home/ubuntu/Validation_data'

EPOCH_NUMBER = 20
BATCH_SIZE = 32

#We build the dataset.
datagen = keras.preprocessing.image.ImageDataGenerator(shear_range = 0.2,
                                                       zoom_range = 0.2,
                                                       horizontal_flip = True,
                                                       width_shift_range=0.2,
                                                       height_shift_range=0.2,
                                                       rotation_range=15,
                                                       vertical_flip=True,
                                                       fill_mode='reflect',
                                                       data_format='channels_last',
                                                       brightness_range=[0.5, 1.5])
train_set = datagen.flow_from_directory(path_t, class_mode = 'binary', batch_size = BATCH_SIZE)
validation_set = datagen.flow_from_directory(path_v, class_mode = 'binary', batch_size = BATCH_SIZE)

#We build the model.
model = keras.Sequential()

model.add(layer.Conv2D(3, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(6, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(12, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25)) #different interpretation
model.add(layer.Conv2D(3, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(6, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(12, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))

model.add(layer.Flatten())
model.add(layer.Dense(64, activation = 'relu'))
#model.add(Dropout(0.25)) #different interpretation
model.add(layer.Dense(32, activation = 'relu'))
model.add(layer.Dense(1, activation = 'sigmoid'))

# Run the model

model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit_generator(train_set, verbose = 1, epochs = EPOCH_NUMBER, validation_data = validation_set)
#score = model.evaluate_generator(validation_set)
#print('Test loss:', score[0])
#print('Test acc:', score[1])

# Plot training & validation accuracy values|| Borrowed from keras documentation.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Accuracy.png")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Loss.png")
