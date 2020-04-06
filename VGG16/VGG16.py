import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation

class VGG16:
    @staticmethod
    def build(width, height, depth, classes):
        #inisiasi model
        model = Sequential()
        inputShape = (height, width, depth)

        #1
        model.add(Conv2D(64, (3,3), padding="same", input_shape = inputShape))
        model.add(Activation("relu"))

        #2
        model.add(Conv2D(64, (3,3), padding ="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

        #3
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))

        #4
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #5
        model.add(Conv2D(256, (3,3), padding="same"))
        model.add(Activation("relu"))

        #6
        model.add(Conv2D(256, (3,3), padding="same"))
        model.add(Activation("relu"))

        #7
        model.add(Conv2D(256, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #8
        model.add(Conv2D(512, (3,3), padding="same"))
        model.add(Activation("relu"))

        #9
        model.add(Conv2D(512, (3,3), padding="same"))
        model.add(Activation("relu"))

        #10
        model.add(Conv2D(512, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #11
        model.add(Conv2D(512, (3,3), padding="same"))
        model.add(Activation("relu"))

        #12
        model.add(Conv2D(512, (3,3), padding="same"))
        model.add(Activation("relu"))

        #13
        model.add(Conv2D(512, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #14fC
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # model.add(Dropout(0.5))

        #15fc
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # model.add(Dropout(0.5))

        #16Softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return model
        return model





    