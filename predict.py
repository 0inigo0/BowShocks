import tensorflow as tf
from tensorflow import keras
from keras.initializers import glorot_uniform
from keras.models import load_model
from keras.models import model_from_json
import json
from keras.preprocessing import image
import numpy as np

#GLOBAL VARIABLES
path_t = '/home/ubuntu/Training_data'
path_v = '/home/ubuntu/BowShocks/Validation_data'

EPOCH_NUMBER = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0005 # its 0.01 by default if we just specify 'adam'.

#We build the dataset.
datagen = keras.preprocessing.image.ImageDataGenerator()
train_set = datagen.flow_from_directory(path_t, class_mode = 'binary', batch_size = BATCH_SIZE)
validation_set = datagen.flow_from_directory(path_v, class_mode = 'binary', batch_size = BATCH_SIZE)

#load model#
json_file = open('/home/ubuntu/BowShocks/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("/home/ubuntu/BowShocks/model_1.h5")
optimizer =keras.optimizers.Adam(learning_rate = 0.0005, beta_1 = 0.9, beta_2 = 0.999, amsgrad=False)
loaded_model.compile(optimizer = optimizer,loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
score = loaded_model.evaluate_generator(validation_set)
print(score)

