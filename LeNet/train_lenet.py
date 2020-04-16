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
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from lenet import LeNet
import matplotlib.pyplot as plt
import random
import os
import argparse


np.seterr(divide='ignore', invalid='ignore')

#konstruk argumen
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
ap.add_argument("-d", "--plot2", type=str, default="plotAkurasi.png",
	help="path to output accuracy/loss plot") 
args = vars(ap.parse_args())

# inisiasi variabel
EPOCHS = 20
#10^-3
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
    image = cv2.resize(image,(32,32))
    image = img_to_array(image)
    data.append(image)

    print("Tingkat Roasting:", make, "Gambar :", imagePath, "Iteration: ",iter)
    iter+=1

    if make == "Mentah ":
        label = 0
    elif make == "Light Roast ":
        label = 1
    elif make =="Medium Roast ":
        label = 2
    elif make == "Dark Roast ":
        label = 3     
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
model = LeNet.build(width=32, height=32, depth=3, classes=4)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
# opt = Nadam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the network
print("[Info] training model")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, validation_steps=len(testX) // BS,
    epochs= EPOCHS, verbose=1)

#save model
print("[Info] menyimpan model")
model.save('train_lenet.model.h5')
# model.save('train_lenet(SGD).model.h5')
# model.save('train_lenet(NADAM).model.h5')

#simpan model ke disk
# model.save(args["model"])

#encoding pada labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
#confusion matrix
print("[Info] Evaluasi Network......")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print("[Info] Classification Report")
print (classification_report(testY.argmax(axis=1), predIdxs, 
        target_names=[str(x) for x in lb.classes_]))

#accuracy, sensivitas, spesifitas
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
print("[Info] Data uji - Data Hasil Predict")
print(testY.argmax(axis=1))
print(predIdxs)
total = sum(sum(cm))
print("total",total)
# Akurasi
acc = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3]) / total

#Spesifitas
spesivitasME = (cm[1,1] + cm[2,2] + cm[3,3]) / ((cm[1,1] + cm[2,2] + cm[3,3]) + (cm[0,1] + cm[0,2] + cm[0,3]))
spesivitasLR = (cm[0,0] + cm[2,2] + cm[3,3]) / ((cm[0,0] + cm[2,2] + cm[3,3]) + (cm[0,1] + cm[0,2] + cm[0,3]))
spesivitasMR = (cm[0,0] + cm[1,1] + cm[3,3]) / ((cm[0,0] + cm[1,1] + cm[3,3]) + (cm[0,1] + cm[0,2] + cm[0,3]))
spesivitasDR = (cm[0,0] + cm[1,1] + cm[2,2]) / ((cm[0,0] + cm[1,1] + cm[2,2]) + (cm[0,1] + cm[0,2] + cm[0,3]))

#Sensivitas
sensivitasME = cm[0,0] / (cm[0,0] + (cm[1,0] + cm[2,0] + cm[3,0]))
sensivitasLR = cm[1,1] / (cm[1,1] + (cm[0,1] + cm[2,1] + cm[3,1]))
sensivitasMR = cm[2,2] / (cm[2,2] + (cm[0,2] + cm[2,2] + cm[3,2]))
sensivitasDR = cm[3,3] / (cm[3,3] + (cm[0,3] + cm[1,3] + cm[2,3]))

print("[Info] Confusion Matrix")
print(cm)
print("akurasi (rumus sendiri): {:.2f}".format(acc))
print("Sensivitas Light Roast: {:.2f}".format(sensivitasLR))
print("Sensivitas Medium Roast: {:.2f}".format(sensivitasMR))
print("Sensivitas Dark Roast: {:.2f}".format(sensivitasDR))
print("Sensivitas Mentah: {:.2f}".format(sensivitasME))

print("Spesivitas Light Roast: {:.2f}".format(spesivitasLR))
print("Spesivitas Medium Roast: {:.2f}".format(spesivitasMR))
print("Spesivitas Dark Roast: {:.2f}".format(spesivitasDR))
print("Spesivitas Mentah: {:.2f}".format(spesivitasME))

#plot training loss dan accuracy
plt.style.use("ggplot")
plt.figure()
N =  EPOCHS
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss Tingkatan Roasting Kopi")
plt.xlabel("20 Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

plt.style.use("ggplot")
plt.figure()
N =  EPOCHS
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Accuracy Tingkatan Roasting Kopi")
plt.xlabel("20 Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot2"])