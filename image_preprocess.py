# 1. Import the necessary packages
import numpy as np
import cv2
from PIL import Image
import webcolors
import time, os, glob
import sys
import shutil
import random

def preprocess_img(filepath):
    img = cv2.imread(filepath, -1)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((31,31), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 15)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    width = 512
    height = 512
    dim = (width, height)
    resized = cv2.resize(result_norm, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray

def crack_detection_preprocess_image():
    # define folders for input data and output data
    folders = ['./YE358311_Fender_apron', './data']
    # DATA_FOLDER = 'YE358311_Fender_apron/defect1'
    DATA_FOLDER_DEFECT_HEALTHY = [folders[0] + '/YE358311_defects/YE358311_Crack_and_Wrinkle_defect', folders[0] + '/YE358311_Healthy']
    DEST_FOLDER = [folders[1] + '/train/defect', folders[1] + '/train/normal']
    # DATA_FOLDER = '/Users/akshyasingh/Documents/coding_practice_python/Defect-Detection-Classifier/data/normal1'
    # DEST_FOLDER = '/Users/akshyasingh/Documents/coding_practice_python/Defect-Detection-Classifier/data/normal1_metal'

    if folders[0] is None or folders[1] is None:
      print "Bad parameters. Please specify input file path and output file path"
      exit()

    current_dir =  os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(current_dir, DATA_FOLDER_DEFECT_HEALTHY[0])
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.jpg')]
    no_defect_files = len(files_txt)
    # no_defect_files = 11

    current = 0
    for DATA_FOLDER in DATA_FOLDER_DEFECT_HEALTHY:
        print "\n\n\nEntering data folder ", DATA_FOLDER[2:]
        count = 1
        current_dir =  os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(current_dir, DATA_FOLDER)
        files = os.listdir(path)
        files_txt = [i for i in files if i.endswith('.jpg')]
        for filename in files_txt:
            if count > no_defect_files:
                print "Successfully pre-processsed %d files" % (2*(count-1))
                break

            current_dir =  os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(current_dir, DATA_FOLDER[2:], filename)
            print "\n\nReading file : ", filename

            # call image preprocess
            gray = preprocess_img(filepath)

            if not os.path.exists(os.path.join(current_dir, DEST_FOLDER[current][2:])):
                os.makedirs(os.path.join(current_dir, DEST_FOLDER[current][2:]))
            savepath = os.path.join(current_dir, DEST_FOLDER[current][2:], str(count) + '.jpg')
            print "File saving in ", savepath
            cv2.imwrite(savepath, gray)
            count += 1
        current  += 1

    DEST_FOLDER = [folders[1] + '/test/defect', folders[1] + '/test/normal']
    if not os.path.exists(os.path.join(current_dir, DEST_FOLDER[0][2:])):
        os.makedirs(os.path.join(current_dir, DEST_FOLDER[0][2:]))
    if not os.path.exists(os.path.join(current_dir, DEST_FOLDER[1][2:])):
        os.makedirs(os.path.join(current_dir, DEST_FOLDER[1][2:]))
    if not os.path.exists(os.path.join(current_dir, 'uploads')):
        os.makedirs(os.path.join(current_dir, 'uploads'))

    return count-1


def move_files_test_folder(no_files=3):
    folders = ['train/normal', 'train/defect']
    folders_dest = ['test/normal', 'test/defect']
    current_dir =  os.path.abspath(os.path.dirname(__file__))

    name_files_to_be_moved = []
    for i in range(2):
        print '\n\nFor i = ', i
        path = os.path.join(current_dir, 'data', folders[i])
        if i == 0:
            to_be_moved = random.sample(glob.glob(path + '/*.jpg'), no_files)
            # print(to_be_moved)
        else :
            for file in to_be_moved:
                file = file.replace('normal', 'defect')
                name_files_to_be_moved.append(file)
            to_be_moved = []
            to_be_moved = name_files_to_be_moved
            # print(to_be_moved)

        for f in enumerate(to_be_moved, 1):
            new_f = f[1].replace('train', 'test')
            dest = os.path.join(current_dir, 'data', folders_dest[i], new_f)
            # if not os.path.exists(dest):
                # os.makedirs(dest)
            shutil.copy(f[1], dest)
            # print('\nsource file moved to : ', dest)
            os.remove(f[1])
            # print('source file deleted from  : ', f[1])
