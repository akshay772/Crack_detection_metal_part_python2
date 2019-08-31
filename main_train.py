from flask import Flask, render_template, request
from werkzeug import secure_filename
from urllib.request import urlopen
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, model_from_json

import numpy as np
import cv2
import urllib
import webcolors
import time, os
import sys, h5py
import shutil
import random

np.random.seed(0)

from image_preprocess import crack_detection_preprocess_image
from image_preprocess import move_files_test_folder
from image_preprocess import preprocess_img
from CNNclassifier import training

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)

# root
@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return '''This is training API :
    To start training, go to http://127.0.0.1:5000/crack_detection_train'''

@app.route('/crack_detection_train')
def train_cnn():
    no_files = crack_detection_preprocess_image()
    # print(type(no_files), no_files)
    to_move = float(no_files)*0.2
    # print(int(to_move), type(to_move))
    move_files_test_folder(int(to_move))
    training()

    return 'Training successful'


if __name__ == '__main__':
   app.run(debug=False)
