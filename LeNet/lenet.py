import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
# from keras.layers.core import Dense, Flatten, Conv2D, MaxPooling2D, Activation


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        #inisiasi model
        model = Sequential()
        inputShape = (height, width, depth)

        # if tf.keras.image_data_format() == "channels_first":
        #     inputShape = (depth, height, width)
        
        #same = zero padding set conv-relu-pool layer 20 layer convolution 5x5
        model.add(Conv2D(20, (5,5), padding="same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

        #set kedua
        model.add(Conv2D(50, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #fully connected layer 
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return model  
        return model
