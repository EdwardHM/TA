import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout

class miniVgg:
    @staticmethod
    def build(width, height, depth, classes):
        #inisiasi model
        model = Sequential()
        inputShape = (height, width, depth)

        #1
        model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))

        #2
        model.add(Conv2D(32, (3,3), padding = "same" ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        model.add(Dropout(0.5))

        #3
        model.add(Conv2D(64, (3,3), padding = "same"))
        model.add(Activation("relu"))

        #4
        model.add(Conv2D(64, (3,3), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
        model.add(Dropout(0.5))

        #5 FC
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        #6
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return model
        return model
