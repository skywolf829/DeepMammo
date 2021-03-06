import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import scipy
import cv2
import os
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import utility_functions
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_val_predict, RepeatedKFold

image_names_train = []
image_names_test = []
labels_train = []
labels_test = []
feature_vectors = []
feature_vectors_test = []
filtered_imgs=[]
filters = []

def find_min_dist_feat(feat_vec):
    min_dist = np.inf
    min_spot = None
    for i in range(len(feature_vectors)):
        d = 0
        for j in range(len(feature_vectors[i])):
            d = d + abs(feat_vec[j] - feature_vectors[i][j])
        if d < min_dist:
            min_dist = d
            min_spot = i
    return min_spot    

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Set up the kernels (24 of them with different size and orientation)
for i in range(6):
    for j in range(4):
        g_kernel = cv2.getGaborKernel((11 + 5*j, 11 + 5*j), 8.0, i*np.pi/5, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filters.append(g_kernel)

# Fill feature dict with normal training images
for item in os.listdir("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTrain"):
    img = cv2.imread(os.path.join("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTrain", item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_names_train.append(item)
    labels_train.append(0)
    feat_vec = []
    for i in range(len(filters)):
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        feat_vec.append(filtered_img.mean())
        feat_vec.append(filtered_img.std())
    feature_vectors.append(feat_vec)

# Fill feature dict with cancerous training images
for item in os.listdir("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTrain"):
    img = cv2.imread(os.path.join("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTrain", item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_names_train.append(item)
    labels_train.append(1)
    feat_vec = []
    for i in range(len(filters)):
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        feat_vec.append(filtered_img.mean())
        feat_vec.append(filtered_img.std())
    feature_vectors.append(feat_vec)

feature_vectors = np.array(feature_vectors)
labels_train = np.array(labels_train)

C_values = [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
C_value_scores = []
clf = LinearSVC()
for spot in range(len(C_values)):
    clf = LinearSVC(C=C_values[spot])
    kFolds = 5
    iterations = 1000
    random_state = 4597834
    i = 0
    averageScore = 0
    rollingAverage = 0
    rkf = RepeatedKFold(n_splits=kFolds, n_repeats=iterations, random_state=random_state)

    for train_index, test_index in rkf.split(feature_vectors):
        X_train, X_test = feature_vectors[train_index], feature_vectors[test_index]
        y_train, y_test = labels_train[train_index], labels_train[test_index]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        averageScore = averageScore + score
        rollingAverage = rollingAverage + score
        i = i + 1
        if i % kFolds == 0:
            #print("Average for " + str(kFolds) + "-split " + str(i / kFolds) + ": " + str (rollingAverage / kFolds))
            rollingAverage = 0

    averageScore = averageScore / i

    print("Average score: " + str(averageScore))
    C_value_scores.append(averageScore)
print(C_value_scores)
quit()
for item in os.listdir("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTest"):
    img = cv2.imread(os.path.join("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTest", item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_names_test.append(item)
    feat_vec = []
    for i in range(len(filters)):
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        feat_vec.append(filtered_img.mean())
        feat_vec.append(filtered_img.std())
    feature_vectors_test.append(feat_vec)
    

for item in os.listdir("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTest"):
    img = cv2.imread(os.path.join("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTest", item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_names_test.append(item)
    feat_vec = []
    for i in range(len(filters)):
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        feat_vec.append(filtered_img.mean())
        feat_vec.append(filtered_img.std())
    feature_vectors_test.append(feat_vec)

labels_test = clf.predict(feature_vectors_test)
confidence_values = clf.decision_function(feature_vectors_test)

utility_functions.printListInOrder(image_names_test)
print("break")
utility_functions.printListInOrder(labels_test)