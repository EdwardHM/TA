import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.layers.core import Dropout

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes):
        #inisiasi model
        model = Sequential()
        inputShape = (height, width, depth)
        # chandim = -1

        # Block1 conv ke-1 > RELU > Pool Layer Set
        model.add(Conv2D(96,(11, 11), strides=(4,4), input_shape=inputShape,
                  padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chandim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # model.add(Dropout(0.25))

        # Block2 conv ke-2 > RELU > Pool Layer Set
        model.add(Conv2D(256,(5,5), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chandim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # model.add(Dropout(0.25))

        #Block3 conv ke-3 > RELU > conv ke-4 > RELU > conv ke-5 > RELU > Pool Layer Set
        model.add(Conv2D(384, (3,3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chandim))

        model.add(Conv2D(384, (3,3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chandim))

        model.add(Conv2D(256, (3,3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chandim))

        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # model.add(Dropout(0.25))

        #block 4 FC Ke 1 >RELU
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        #block 5 FC ke 2
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        #softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return model
        return model

  






