import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

#konstruk argumen
# ap = argparse.ArgumentParser()
# # ap.add_argument("-m", "--model", required=True,
# # 	help="path to trained model model")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

#load image 
path = "../LuarSet/dark_roast (2).jpg"
image = cv2.imread(path, cv2.IMREAD_COLOR)
orig = image.copy()

#prepocessing histogram eq rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(image)

# equalize Hist operation di Y channel.
y_eq = cv2.equalizeHist(y)

image = cv2.merge((y_eq, cr, cb))
image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)

#resize
image = cv2.resize(image,(227,227))
# cv2.imshow("o",image)
# cv2.waitKey(0)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

#load model yang sudah dilatih
print("[Info] Loading Network.....")
# model = load_model("train_lenet.model.h5")
model = tf.keras.models.load_model("train_AlexNet.model.h5")
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

# # draw label di image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# # show output image
cv2.imshow("Output", output)
cv2.waitKey(0)