from flask import Flask, render_template     
from imutils import paths 
import os

app = Flask(__name__)

@app.route("/")
def home():
    images = os.listdir(os.path.join(app.static_folder, "Gambar_Send"))
    # imagePaths = sorted(list(paths.list_images("../prepo")))
    return render_template("Home.html", images=images)
    
@app.route("/salvador")
def salvador():
    return "Hello, Salvador"
    
if __name__ == "__main__":
    app.run(debug=True)
