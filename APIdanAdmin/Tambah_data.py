from flask import Flask, render_template, request   
from imutils import paths 
import json
import os
import cv2
from imageio import imread
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

@app.route("/")
def home():
    images = os.listdir(os.path.join(app.static_folder, "Gambar_Send"))
    # imagePaths = sorted(list(paths.list_images("../prepo")))
    return render_template("Home.html", images=images)
    
@app.route("/Move", methods=["GET", "POST"])
def Move():
    data = request.get_json('gambar')
    namaAwal = str(data["gambar"])
    # kelas = request.get_json('kls')
    kelas = str(data["kls"])
    path = "E:/Kuliah/TA(Program)/APIdanAdmin/static/Gambar_Send/"+namaAwal
    pindah = "E:/Kuliah/TA(Program)/ujiPindah"
    img_data = cv2.imread(path, cv2.IMREAD_COLOR)
    # histogram equalization RGB
    img_y_cr_cb = cv2.cvtColor(img_data, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # equalize Hist operation di Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    #simpan ke prepo
    #ambil nama asli file e
    savename = kelas+".jpg"
    cv2.imwrite(os.path.join(pindah, savename),img_rgb_eq)
    # cv2.imshow('Test image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    response = "Response: Gambar: " + namaAwal + " Menjadi: "+kelas
    return response
    
if __name__ == "__main__":
    app.debug = True
    app.run(host='192.168.1.6', port=5001)
