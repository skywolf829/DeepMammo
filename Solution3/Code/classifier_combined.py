import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import pickle
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_val_predict, RepeatedKFold
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
import argparse
import tensorflowvgg.vgg19 as vgg19
import utility_functions
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

codes_path = './codes'
labels_path = './labels'
names_path = './names'
radio_input_classify, radio_input_confidence = utility_functions.loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)


images_normal_train, labels_normal_train, names_normal_train = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/NormalTrain",), (0,))
images_normal_test, labels_normal_test, names_normal_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/NormalTest",), (0,))
images_abnormal_train, labels_abnormal_train, names_abnormal_train = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/AbnormalTrain",), (1,))
images_abnormal_test, labels_abnormal_test, names_abnormal_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/AbnormalTest",), (1,))
images_contralateral_test, labels_contralateral_test, names_contralateral_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/ContralateralTest",), (0,))


sess = tf.Session()
print("Session start")

vgg = vgg19.Vgg19()
input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
with tf.name_scope("content_vgg"):
    vgg.build(input_)
# Get the values from the relu6 layer of the VGG network
feed_dict_normal_train = {input_: images_normal_train}
feed_dict_normal_test = {input_: images_normal_test}
feed_dict_cancer_train = {input_: images_abnormal_train}
feed_dict_cancer_test = {input_: images_abnormal_test}
feed_dict_contralateral = {input_: images_contralateral_test}

codes_normal_train = sess.run(vgg.relu6, feed_dict=feed_dict_normal_train)
codes_normal_test = sess.run(vgg.relu6, feed_dict=feed_dict_normal_test)
codes_cancer_train = sess.run(vgg.relu6, feed_dict=feed_dict_cancer_train)
codes_cancer_test = sess.run(vgg.relu6, feed_dict=feed_dict_cancer_test)
codes_contralateral = sess.run(vgg.relu6, feed_dict=feed_dict_contralateral)
sess.close()

clf = LinearSVC(C=0.0001)

X_train = np.append(codes_normal_train, codes_cancer_train, axis=0)
X_test = np.append(codes_normal_test, codes_cancer_test, axis=0)

y_train = np.append(labels_normal_train, labels_abnormal_train, axis=0)
y_test = np.append(labels_normal_test, labels_abnormal_test, axis=0)

names_train = np.append(names_normal_train, names_abnormal_train, axis=0)
names_test = np.append(names_normal_test, names_abnormal_test, axis=0)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
confidence_values = clf.decision_function(X_test)
confidence_values_contralateral = clf.decision_function(codes_contralateral)
predictions_contralateral = clf.predict(codes_contralateral)

model_confidence = {}
model_confidence_train = {}
model_classification = {}
model_classification_train = {}
model_classification_contralateral = {}
model_confidence_contralateral = {}

i = 0
for item in confidence_values:
    model_confidence[names_test[i]] = abs(item)
    i = i + 1

i = 0
for item in clf.decision_function(X_train):
    model_confidence_train[names_train[i]] = abs(item)
    i = i + 1

i = 0
for item in predictions:
    model_classification[names_test[i]] = item
    i = i + 1

i = 0
for item in clf.predict(X_train):
    model_classification_train[names_train[i]] = item
    i = i + 1

i = 0
for item in predictions_contralateral:
    model_classification_contralateral[names_contralateral_test[i]] = item
    i = i + 1

i = 0
for item in confidence_values_contralateral:
    model_confidence_contralateral[names_contralateral_test[i]] = abs(item)
    i = i + 1


combined_inputs_X_train = []
combined_inputs_y_train = []
combined_inputs_X_test = []
combined_inputs_y_test = []

for i in range(len(names_train)):
    newItem = []
    if names_train[i] in radio_input_classify.keys():
        newItem.append(model_classification_train[names_train[i]])
        newItem.append(model_confidence_train[names_train[i]])
        newItem.append(radio_input_classify[names_train[i]])
        newItem.append(radio_input_confidence[names_train[i]])
        combined_inputs_X_train.append(newItem)
        combined_inputs_y_train.append(y_train[i])

for i in range(len(names_test)):
    newItem = []
    if names_test[i] in radio_input_classify.keys():
        newItem.append(model_classification[names_test[i]])
        newItem.append(model_confidence[names_test[i]])
        newItem.append(radio_input_classify[names_test[i]])
        newItem.append(radio_input_confidence[names_test[i]])
        combined_inputs_X_test.append(newItem)
        combined_inputs_y_test.append(y_test[i])

clf = MLPClassifier(hidden_layer_sizes=(3), activation="logistic", max_iter=1000000)

clf.fit(combined_inputs_X_train, combined_inputs_y_train)
predictions = clf.predict(combined_inputs_X_test)
score = clf.score(combined_inputs_X_test, combined_inputs_y_test)

print("Training set size: " + str(len(combined_inputs_X_train)))
print("Test set size: " + str(len(combined_inputs_X_test)))
print("Final score: " + str(score))
utility_functions.printListInOrder(names_test)
print("break")
utility_functions.printListInOrder(predictions)
#print("break")
#utility_functions.printDictionaryInOrder(names_contralateral_test, model_confidence_contralateral)


