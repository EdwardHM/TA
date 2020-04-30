from flask import Flask, request #import main Flask class and request object
import json
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import base64
import uuid
import os
import io
from imageio import imread

app = Flask(__name__) #create the Flask app

@app.route('/LeNet-Adam', methods=['POST']) 
def LeNet_Adam():
    
    #Setting for using CPU
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.list_logical_devices('GPU')

    url = request.get_json('imgBs64')
    # print(url)
    stri1= str(url)[0:].split(" ")[1]
    stri2= stri1[0:].split("'")[1]
    stri3= stri2[0:].split("'")[0]
    img = imread(io.BytesIO(base64.b64decode(stri3)))
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(cv2_img)
    path = "E:/Kuliah/TA(Program)/API/Gambar_Send"
    nama = str(uuid.uuid4())+".jpg"
    cv2.imwrite(os.path.join(path, nama), cv2_img)
    pathimage = "E:/Kuliah/TA(Program)/API/Gambar_Send/"+nama
    print(nama)
    # path = "D:/XAMPP/htdocs/API_TA/dark_roast(1).jpg"
    image = cv2.imread(pathimage, cv2.IMREAD_COLOR)
    orig = image.copy()
    # print(stri1)
    # print(stri2)
    # print(stri3)

    #prepocessing histogram eq rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)

    # equalize Hist operation di Y channel.
    y_eq = cv2.equalizeHist(y)

    image = cv2.merge((y_eq, cr, cb))
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)

    #resize
    image = cv2.resize(image,(32,32))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    #load model yang sudah dilatih
    print("[Info] Loading Network.....")
    # model = load_model("train_lenet.model.h5")
    model = tf.keras.models.load_model("train_lenet.model.h5")
    array = model.predict(image)
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        label = "Mentah"
    elif answer == 1:
        label = "Light Roast"
    elif answer == 2:
        label = "Medium Roast"
    elif answer == 3:
        label = "Dark Roast"

    label = "{}".format(label)

    print("hasil prediksi : " + label)
    return label

@app.route('/LeNet-Nadam', methods=['POST']) 
def LeNet_Nadam():
    
    #Setting for using CPU
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.list_logical_devices('GPU')

    url = request.get_json('imgBs64')
    stri1= str(url)[0:].split(" ")[1]
    stri2= stri1[0:].split("'")[1]
    stri3= stri2[0:].split("'")[0]
    img = imread(io.BytesIO(base64.b64decode(stri3)))
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(cv2_img)
    path = "E:/Kuliah/TA(Program)/API/Gambar_Send"
    nama = str(uuid.uuid4())+".jpg"
    cv2.imwrite(os.path.join(path, nama), cv2_img)
    pathimage = "E:/Kuliah/TA(Program)/API/Gambar_Send/"+nama
    print(nama)
    # path = "D:/XAMPP/htdocs/API_TA/dark_roast(1).jpg"
    image = cv2.imread(pathimage, cv2.IMREAD_COLOR)
    orig = image.copy()
    # print(stri1)
    # print(stri2)
    # print(stri3)

    #prepocessing histogram eq rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)

    # equalize Hist operation di Y channel.
    y_eq = cv2.equalizeHist(y)

    image = cv2.merge((y_eq, cr, cb))
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)

    #resize
    image = cv2.resize(image,(32,32))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    #load model yang sudah dilatih
    print("[Info] Loading Network.....")
    # model = load_model("train_lenet.model.h5")
    model = tf.keras.models.load_model("train_lenet(NADAM).model.h5")
    array = model.predict(image)
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        label = "Mentah"
    elif answer == 1:
        label = "Light Roast"
    elif answer == 2:
        label = "Medium Roast"
    elif answer == 3:
        label = "Dark Roast"

    label = "{}".format(label)

    print("hasil prediksi : " + label)
    return label

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.7')