import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

# basic
# img = cv2.imread("../Images/Medium_Roast/Medium_Roast (56).jpg",1)
# img_resize =  cv2.resize(img, (224, 224)) 
# cv2.imshow('image awal',img_resize)
# plt.hist(img.ravel(),256,[0,256])
# plt.title("Gambar Awal")
# plt.show()

# buat dapetin histogram tiap warna
# color = ('b', 'g', 'r')
# for channel,col in enumerate(color):
#     histr = cv2.calcHist([img], [channel], None, [256], [0,256])
#     plt.plot(histr,color = col)
#     plt.title('Histogram for color scale image')
#     plt.show()
# while True:
#     k = cv2.waitKey(0) & 0xFF
#     if k==27: break
# cv2.destroyAllWindows()

# img_y_cr_cb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2YCrCb)
# y, cr, cb = cv2.split(img_y_cr_cb)

# # equalize Hist operation di Y channel.
# y_eq = cv2.equalizeHist(y)

# img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
# img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

# cv2.imshow("histogram eq", img_rgb_eq)
# plt.hist(img_rgb_eq.ravel(),256,[0,256])
# plt.title("Gambar Equalization")
# plt.show()
# cv2.waitKey(0)


i = 1

Train_dir = "../Dataset"
Prepo_dir = "../prepo"

# baca semua yang ada di dataset
for img in tqdm(os.listdir(Train_dir)):
    img_path = os.path.join(Train_dir, img)
    img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # histogram equalization RGB
    img_y_cr_cb = cv2.cvtColor(img_data, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # equalize Hist operation di Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    #simpan ke prepo
    # counter = str(i)
    # nama = ""+counter+".jpg"

    #ambil nama asli file e
    nama = ""+img_path[10:].split(".")[0]
    savename = os.path.basename(nama)+".jpg"
    # print(nama)
    cv2.imwrite(os.path.join(Prepo_dir, savename),img_rgb_eq)
    # i +=1
    cv2.waitKey(0)

