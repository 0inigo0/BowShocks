import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image1 = image.load_img(filename)
    width, height = image.size
    # load the image with the required size
    image1 = image.load_img(filename, target_size=shape)
    # convert to numpy array
    image1 = image.img_to_array(image1)
    # scale pixel values to [0, 1]
    image1 = image1.astype('float32')
    image1 /= 255.0
    # add a dimension so that we have one sample
    image1 = expand_dims(image1, 0)
    return image1, width, height
 
input_w, input_h = 300, 300
# define our new photo
photo_filename = 'Bow_shock.png'
# load and prepare image
image1, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

#load model#
model = load_model('model_1.h5')
yhat = model.predict(image1)
