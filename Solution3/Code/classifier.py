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
import utility_functions

cancer_images = '../Images/CANCER'
contralateral_cancer_images = '../Images/CONTRALATERAL BREAST TO CANCEROUS'
normal_images = '../Images/NORMAL'
codes_path = './codes'
labels_path = './labels'

class_0 = contralateral_cancer_images
class_1 = normal_images

images, labels = utility_functions.loadImagesFromDir((class_0, class_1), (0,1))
print("Images: " + str(np.shape(images)))
print("Labels: " + str(np.shape(labels)))

sess = tf.Session()
print("Session start")

vgg = vgg19.Vgg19()
input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
with tf.name_scope("content_vgg"):
    vgg.build(input_)
# Get the values from the relu6 layer of the VGG network
feed_dict = {input_: images}
codes = sess.run(vgg.relu6, feed_dict=feed_dict)
sess.close()

np.save(codes_path, codes) 
np.save(labels_path, labels) 

pickle.dump(codes, open('codes', 'wb'))
pickle.dump(labels, open('labels', 'wb'))
print("Codes loaded: " + str(np.shape(codes)))
print("Labels loaded: " + str(np.shape(labels)))

X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.2, random_state=0)

clf = LinearSVC(C=0.0001)
clf.fit(X_train, y_train)

print("Confidence values")
print(clf.decision_function(X_test))

print("Predictions")
print(clf.predict(X_test))

print("Actual values")
print(y_test)

print("Accuracy score")
print(clf.score(X_test, y_test))