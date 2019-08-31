from flask import Flask, render_template, request
from werkzeug import secure_filename
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, model_from_json

import tensorflow as tf
import numpy as np
import cv2
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

# load json and create model
json_file = open('./models/model_92.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load model and weights
loaded_model = model_from_json(loaded_model_json)
# loaded_model = load_model("./models/first_try.h5")
loaded_model.load_weights("./models/first_try_92.h5")
# print(loaded_model.summary())
print "Loaded model from disk"

graph = tf.get_default_graph()

# root
@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return '''This is testing API :
    To start predicting, go to http://127.0.0.1:5000/crack_detection_test'''

@app.route('/crack_detection_test', methods = ['GET','POST'])
def crack_detection_test():
    class_pred = ''
    filenames = ''
    filename = ''
    # test for input file
    if request.method =='POST':
        file = request.files['file[]']
        # print(file)
        if file:
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            filename = secure_filename(file.filename)
            if not os.path.exists(os.path.join(current_dir, 'uploads')):
                os.makedirs(os.path.join(current_dir, 'uploads'))
            filepath = os.path.join(current_dir, 'uploads', filename)
            file.save(filepath)

        # print(filepath)
        img = cv2.imread(filepath)
        gray = preprocess_img(filepath)

        # new_im = Image.fromarray(gray)
        # new_im.show()

        # save the test image in test_dir
        if not os.path.exists(os.path.join(current_dir, 'uploads', 'test', 'images')):
            os.makedirs(os.path.join(current_dir, 'uploads', 'test', 'images'))
        test_dir = os.path.join(current_dir, 'uploads', 'test')
        # save the test file to test directory
        savepath = os.path.join(test_dir, 'images', filename)
        # print(filename)
        # print(savepath)
        cv2.imwrite(savepath, gray)

        try :
            batch_size = 16
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                                test_dir,
                                target_size=(512, 512),
                                batch_size=batch_size,
                                class_mode='binary',
                                shuffle=False)

            #resetting generator
            test_generator.reset()

            with graph.as_default():
                # make predictions
                pred=loaded_model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

                # Get classes by np.round
                cl = np.round(pred)
                # Get filenames (set shuffle=false in generator is important)
                filenames=test_generator.filenames

                if cl[:,0][0] == np.float32(1.0):
                    class_pred = 'Healthy'
                else :
                    class_pred = 'Defective'
                # 0 is defective, 1 is healthy
                print "\nfile : ", filenames ,"\nprediction : ", pred[:,0], "\nclass : ", cl[:,0][0], '\npredicted class : ', class_pred

        except Exception as e:
            print e
            print "Please try again.... something has gone wrong"

        finally :
            # remove all uploaded files
            os.remove(savepath)
            os.remove(filepath)

        # # evaluate loaded model on test data
        # loaded_model.compile(loss='binary_crossentropy',
        #           optimizer='adam',
        #           metrics=['accuracy'])

        # predict = loaded_model.predict(gray)
        # print(predict)

    return render_template('file_upload.html', string_variable= class_pred, image_name= filename)

if __name__ == '__main__':
   app.run(debug=True)
