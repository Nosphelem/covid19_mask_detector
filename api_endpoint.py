# LIBRARIES

import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)

upload_folder = os.getcwd() + "../upload_folder"
allowed_extensions = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["upload_folder"] = upload_folder

def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in allowed_extensions

@app.route("/", methods = ["GET", "POST"])
def upload_file():
    if request.method == "POST"