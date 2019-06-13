import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import cv2
import os
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import utility_functions
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve

image_names_train = []
image_names_test = []
labels_train = []
labels_test = []
predictions = []
feature_vectors = []
filtered_imgs = []
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

for item in os.listdir("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTest"):
    img = cv2.imread(os.path.join("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTest", item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_names_test.append(item)
    labels_test.append(0)
    feat_vec = []
    for i in range(len(filters)):
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        feat_vec.append(filtered_img.mean())
        feat_vec.append(filtered_img.std())
    closest_index = find_min_dist_feat(feat_vec)
    if image_names_train[closest_index][:1] == "N":
        predictions.append(0)
    else:
        predictions.append(1)

for item in os.listdir("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTest"):
    img = cv2.imread(os.path.join("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTest", item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_names_test.append(item)
    labels_test.append(1)
    feat_vec = []
    for i in range(len(filters)):
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
        feat_vec.append(filtered_img.mean())
        feat_vec.append(filtered_img.std())
    closest_index = find_min_dist_feat(feat_vec)
    if image_names_train[closest_index][:1] == "N":
        predictions.append(0)
    else:
        predictions.append(1)

fpr, tpr, thresholds = roc_curve(labels_test, predictions)
roc_auc = auc(fpr, tpr)
print(roc_auc)