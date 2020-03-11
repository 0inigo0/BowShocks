import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layer
from tensorflow.keras.models import model_from_json

#GLOBAL VARIABLES
path_t = '/home/ubuntu/Training_data'
path_v = '/home/ubuntu/Validation_data'

EPOCH_NUMBER = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001 # its 0.01 by default if we just specify 'adam'.

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
'''
model.add(layer.Conv2D(3, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(6, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(12, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25)) #different interpretation
model.add(layer.Conv2D(24, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(36, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(48, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))

model.add(layer.Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(128, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(256, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(320, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))

model.add(layer.Flatten())
model.add(layer.Dense(128, activation = 'relu'))
#model.add(Dropout(0.25)) #different interpretation
model.add(layer.Dense(32, activation = 'relu'))
'''
model.add(layer.Conv2D(8, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(16, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(24, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.25)) #different interpretation
model.add(layer.Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(48, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))

model.add(layer.Conv2D(96, kernel_size = (3,3), activation = 'relu'))
model.add(layer.Conv2D(128, kernel_size = (3,3), activation = 'relu')) #it knows what input shape is
model.add(layer.Conv2D(256, kernel_size = (3,3), activation = 'relu'))
model.add(layer.MaxPooling2D(pool_size = (2,2)))

model.add(layer.Flatten())
model.add(layer.Dense(256, activation = 'relu'))
#model.add(Dropout(0.25)) #different interpretation
model.add(layer.Dense(32, activation = 'relu'))
model.add(layer.Dense(1, activation = 'sigmoid'))


#Create the optimizer: we are using adam.
optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, amsgrad=False)
# Run the model

model.compile(optimizer = optimizer,loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit_generator(train_set, verbose = 1, epochs = EPOCH_NUMBER, validation_data = validation_set)
#score = model.evaluate_generator(validation_set)
#print('Test loss:', score[0])
#print('Test acc:', score[1])

'''Save Model to h5 '''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_1.h5")
print("Saved model to disk")

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


