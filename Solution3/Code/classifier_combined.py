import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import pickle
from os.path import isfile, isdir
import numpy as np
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
import argparse
import utility_functions

with open('radio_input_classify', "rb") as input_file:
    radio_input_classify = pickle.load(input_file)
with open('radio_input_confidence', "rb") as input_file:
    radio_input_confidence = pickle.load(input_file)
with open('model_input_confidence', "rb") as input_file:
    model_decision_confidence = pickle.load(input_file)
with open('model_input_classify', "rb") as input_file:
    model_decision_classification = pickle.load(input_file)
with open('labels', "rb") as input_file:
    labels = pickle.load(input_file)
with open('names', "rb") as input_file:
    names  = pickle.load(input_file)
#print(names)
#print(labels)
#print(radio_input_classify)

inputs = utility_functions.createFeaturesFromDicts(model_decision_confidence, model_decision_classification, radio_input_confidence, radio_input_classify, names)
inputs = np.array(inputs)
labels = np.array(labels)

print(inputs)
print(labels)

#clf = SVC(kernel='linear', gamma='scale', C=1)
#clf = SVC(kernel='rbf', gamma='scale', C=1)
#clf = KNeighborsClassifier(3)
#clf = DecisionTreeClassifier(max_depth=5, max_features=1)
#clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf = MLPClassifier(hidden_layer_sizes=(2), alpha=1, solver="adam", max_iter=100000, tol=0.00000001, verbose=10)
#clf = AdaBoostClassifier()
#clf = GaussianNB()

kFolds = 5
iterations = 50
random_state = 4597834
i = 0
averageScore = 0
rollingAverage = 0
rkf = RepeatedKFold(n_splits=kFolds, n_repeats=iterations, random_state=random_state)

for train_index, test_index in rkf.split(inputs):
    X_train, X_test = inputs[train_index], inputs[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    averageScore = averageScore + score
    rollingAverage = rollingAverage + score
    i = i + 1
    if i % kFolds == 0:
        print("Average for " + str(kFolds) + "-split " + str(i / kFolds) + ": " + str (rollingAverage / kFolds))
        rollingAverage = 0

averageScore = averageScore / i

print("Average score: " + str(averageScore))
