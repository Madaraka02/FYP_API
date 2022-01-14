from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image

# Keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='covid.h5'

# Load your trained model
model = load_model(MODEL_PATH)
class_names = ['Covid', 'Normal']


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    # x=x/255
    x = np.expand_dims(x, 0)

    predictions = model.predict(x)
    score = tf.nn.softmax(predictions[0])
    preds=class_names[np.argmax(score)]
    # preds=class_names[np.argmax(predictions)]

       
    return preds



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)




