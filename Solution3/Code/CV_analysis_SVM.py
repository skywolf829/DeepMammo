import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import pickle
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_val_predict, RepeatedKFold, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve, mean_absolute_error
import argparse
import tensorflowvgg.vgg19 as vgg19
import utility_functions
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import json

codes_path = './codes'
labels_path = './labels'
names_path = './names'
radio_input_classify, radio_input_confidence = utility_functions.loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)


images_normal, labels_normal, names_normal = utility_functions.loadImagesFromDir(("../Images/NewCroppingMethodv5/Normal",), (0,))
images_cancer, labels_cancer, names_cancer = utility_functions.loadImagesFromDir(("../Images/NewCroppingMethodv5/Cancer",), (1,))
names_all = np.append(names_normal, names_cancer, axis=0)
labels_all = np.append(labels_normal, labels_cancer, axis=0)
images_all = np.append(images_normal, images_cancer, axis=0)

sess = tf.Session()
print("Session start")

vgg = vgg19.Vgg19()
input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
with tf.name_scope("content_vgg"):
    vgg.build(input_)
# Get the values from the relu6 layer of the VGG network
feed_dict_all = {input_: images_all}
codes_all = sess.run(vgg.relu6, feed_dict=feed_dict_all)

sess.close()

clf = LinearSVC(C=0.0001)
scores = []
ROCs = []
for iteration in range(100):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=iteration)
    for train_index, test_index in skf.split(codes_all, labels_all):
        X_train, X_test = codes_all[train_index], codes_all[test_index]
        y_train, y_test = labels_all[train_index], labels_all[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))
        roc_auc = auc(fpr, tpr)
        ROCs.append(roc_auc)
        print(str(iteration) + " Score: " + str(scores[len(scores)-1]) + " \t\t\tAUC: " + str(ROCs[len(ROCs)-1]))
print("Avg CV score: " + str(np.average(scores)))
print("STD of CV score: " + str(np.std(scores)))
print("95% CI for CV score: " + str(np.average(scores) - 1.96 * np.std(scores)) + " to " + str(np.average(scores) + 1.96 * np.std(scores)))
print("Avg CV AUC: " + str(np.average(ROCs)))
print("STD of CV AUC: " + str(np.std(ROCs)))
print("95% CI for CV AUC: " + str(np.average(ROCs) - 1.96 * np.std(ROCs)) + " to " + str(np.average(ROCs) + 1.96 * np.std(ROCs)))
