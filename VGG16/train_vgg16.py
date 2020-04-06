from skimage import feature
from imutils import paths
import imutils
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from VGG16 import VGG16
import matplotlib.pyplot as plt
import random
import os
import argparse

# configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
# session = tf.compat.v1.Session(config=configuration)
tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.list_logical_devices('GPU')

np.seterr(divide='ignore', invalid='ignore')

#konstruk argumen
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# inisiasi variabel
EPOCHS = 20
INIT_LR = 1e-3
BS = 32

print("[Info] Membaca Dataset")
labels = []
data = []
iter = 0

#mengambil gambar kemudian diacak
imagePaths = sorted(list(paths.list_images("../prepo")))
random.seed(42)
random.shuffle(imagePaths)


for imagePath in imagePaths:
    make = imagePath[9:].split("(")[0]
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(224,224))
    image = img_to_array(image)
    data.append(image)

    print("Tingkat Roasting:", make, "Gambar :", imagePath, "Iteration: ",iter)
    iter+=1

    if make == "Mentah ":
        label = 0
        # labels.append(label)
    elif make == "Light Roast ":
        label = 1
        # labels.append(label)
    elif make =="Medium Roast ":
        label = 2
        # labels.append(label)
    elif make == "Dark Roast ":
        label = 3
        # labels.append(label)
    labels.append(label)

print(labels)
print("[Info] Selesai Membaca Dataset")

#mengukur intensitas raw pixel dari 0-1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#membagi data menjadi 75% train 25% test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#konversi label dari integer ke vector
trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)

#image generator untuk data augmentation
aug = ImageDataGenerator(rotation_range = 30, width_shift_range=0.1, 
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

#inisiasi model
print("[Info] Compiling Model.....")
model = VGG16.build(width=224, height=224, depth=3, classes=4)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the network
print("[Info] training model")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

#save model
print("[Info] menyimpan model")
model.save('train_VGG16.model.h5')

#simpan model ke disk
# model.save(args["model"])

#plot training loss dan accuracy
plt.style.use("ggplot")
plt.figure()
N =  EPOCHS
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss dan Accuracy Tingkatan Roasting Kopi")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])