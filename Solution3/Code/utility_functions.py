import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import pickle
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import argparse
import tensorflowvgg.vgg19 as vgg19



def loadImagesFromDir(dirs, classification):    
    labels = []
    batch = []
    for dir_i in range(len(dirs)):
        class_x = classification[dir_i]
        dir_x = dirs[dir_i]
        print("Loading class " + str(class_x) + " from " + str(dir_x))
        files = os.listdir(dir_x)
        for i, file in enumerate(files, 1):
            # Add images to the current batch
            #print(i)
            file_name = os.path.join(dir_x, file)
            #print(file_name)
            img = load_img(file_name, target_size=(224, 224))
            img_arr = img_to_array(img)
            #print("Shape of imgarr:")
            #print(img_arr.shape)
            batch.append(img_arr)  # change according to original script
            #print(np.shape(batch))
            labels.append(class_x)
    return batch, labels