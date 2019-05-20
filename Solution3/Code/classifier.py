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

class_0 = cancer_images
class_1 = normal_images

#print(contents)

codes_list = []
codes = None
labels = []
images = []

sess = tf.Session()
print("Session start")
vgg = vgg19.Vgg19()
input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
with tf.name_scope("content_vgg"):
    vgg.build(input_)

images, labels = utility_functions.loadImagesFromDir((class_0, class_1), (0,1))

# Get the values from the relu6 layer of the VGG network
feed_dict = {input_: images}
codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
sess.close()

# Build an array of codes
if codes is None:
    codes = codes_batch
else:
    codes = np.append(codes_batch, codes, axis=0)



np.save(codes_path, codes)  # user input
#print("Shape of codes")
#print(np.shape(codes))
np.save(labels_path, labels)  # user input
#print("Shape of labels")
#print(np.shape(labels))
#print("Labels: " + str(labels))
pickle.dump(codes, open('codes', 'wb'))
pickle.dump(labels, open('labels', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.2, random_state=0)
#print(y_train)
#print(X_test)

clf = LinearSVC(C=0.0001)
# clf = CalibratedClassifierCV(clf)
clf.fit(X_train, y_train)

print("Confidence values")
print(clf.decision_function(X_test))

# print("Prediction probabilities")
# print(clf.predict_proba(X_test))

print("Predictions")
print(clf.predict(X_test))
print("Actual values")
print(y_test)

print("Accuracy score")
print(clf.score(X_test, y_test))
