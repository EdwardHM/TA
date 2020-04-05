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
    image = cv2.resize(image,(32,32))
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

#encoding pada labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)

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
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the network
print("[Info] training model")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, validation_steps=len(testX) // BS,
    epochs= EPOCHS, verbose=1)

#save model
print("[Info] menyimpan model")
model.save('train_lenet.model.h5')

#simpan model ke disk
# model.save(args["model"])

#confusion matrix
# print("[Info] Evaluasi Network......")
# predIdxs = model.predict(testX, batch_size=BS)
# predIdxs = np.argmax(predIdxs, axis=1)
# print (classification_report(testY.argmax(axis=1), predIdxs, 
#         target_names = lb.classes_))

# #accuracy, sensivitas, spesifitas
# cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
# total = sum(sum(cm))
# #Akurasi
# acc = (cm[0,0] + cm[1,1] + cm(2,2) + cm(3,3) ) / total

# #Spesifitas
# spesivitasLR = (cm[1,1]+ cm[2,2], [3,3]) / ((cm[1,1]+ cm[2,2], [3,3]) + (cm[0,1]+cm[0,2]+cm[0,3]))
# spesivitasMR = (cm[0,0]+ cm[2,2], [3,3]) / ((cm[0,0]+ cm[2,2], [3,3]) + (cm[0,1]+cm[0,2]+cm[0,3]))
# spesivitasDR = (cm[0,0]+ cm[1,1], [3,3]) / ((cm[0,0]+ cm[1,1], [3,3]) + (cm[0,1]+cm[0,2]+cm[0,3]))
# spesivitasME = (cm[0,0]+ cm[1,1], [2,2]) / ((cm[0,0]+ cm[1,1], [2,2]) + (cm[0,1]+cm[0,2]+cm[0,3]))

# #Sensivitas
# sensivitasLR = cm[0,0] / (cm[0,0] + (cm[1,0]+cm[2,0]+[3,0]))
# sensivitasMR = cm[1,1] / (cm[1,1] + (cm[1,0]+cm[2,0]+[3,0]))
# sensivitasDR = cm[2,2] / (cm[2,2] + (cm[1,0]+cm[2,0]+[3,0]))
# sensivitasME = cm[3,3] / (cm[3,3] + (cm[1,0]+cm[2,0]+[3,0]))

# print(cm)
# print("akurasi: {:.4f}".format(acc))
# print("Sensivitas Light Roast: {:.4f}".format(sensivitasLR))
# print("Sensivitas Medium Roast: {:.4f}".format(sensivitasMR))
# print("Sensivitas Dark Roast: {:.4f}".format(sensivitasDR))
# print("Sensivitas Mentah: {:.4f}".format(sensivitasME))

# print("Spesivitas Light Roast: {:.4f}".format(spesivitasLR))
# print("Spesivitas Medium Roast: {:.4f}".format(spesivitasMR))
# print("Spesivitas Dark Roast: {:.4f}".format(spesivitasDR))
# print("Spesivitas Mentah: {:.4f}".format(spesivitasME))


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