import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import pickle
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from scipy import stats
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, ShuffleSplit, cross_val_predict, RepeatedKFold, StratifiedKFold, KFold
from sklearn.utils import shuffle, resample
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


images_normal, labels_normal, names_normal = utility_functions.loadImagesFromDir(("../Images/Cropped/Normal",), (0,))
images_cancer, labels_cancer, names_cancer = utility_functions.loadImagesFromDir(("../Images/Cropped/Cancer",), (1,))
# If only using images that have radiologist response
i = 0
while i < len(names_normal):
    names_normal[i] = names_normal[i].split(".")[0] + ".png"
    if names_normal[i] not in radio_input_classify.keys():
        names_normal.pop(i)
        labels_normal.pop(i)
        images_normal.pop(i)
    else:
        i = i + 1
i = 0


while i < len(names_cancer):
    names_cancer[i] = names_cancer[i].split(".")[0] + ".png"
    if names_cancer[i] not in radio_input_classify.keys():
        names_cancer.pop(i)
        labels_cancer.pop(i)
        images_cancer.pop(i)
    else:
        i = i + 1

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
feed_dict_normal = {input_: images_normal}
feed_dict_cancer = {input_: images_cancer}
codes_normal = sess.run(vgg.relu6, feed_dict=feed_dict_normal)
codes_cancer = sess.run(vgg.relu6, feed_dict=feed_dict_cancer)
codes_all = np.append(codes_normal, codes_cancer, axis=0)
sess.close()




clf = LinearSVC(C=0.0001)


# For 10-fold CV
"""
scores = []
ROCs = []
for iteration in range(1000):    
    skf = KFold(n_splits=10, shuffle=True, random_state=iteration)
    for train_index, test_index in skf.split(codes_all, labels_all):
        X_train, X_test = codes_all[train_index], codes_all[test_index]
        y_train, y_test = labels_all[train_index], labels_all[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))
        roc_auc = auc(fpr, tpr)
        ROCs.append(roc_auc)
        print(str(iteration) + " AUC: " + str(ROCs[len(ROCs)-1]))

                       
print("Avg CV score: " + str(np.average(scores)))
print("STDev of CV score: " + str(np.std(scores)))
print("StdErr of CV score: " + str(stats.sem(scores)))
print("95% CI for CV score: " + str(np.average(scores) - 1.96 * stats.sem(scores)) + " to " + str(np.average(scores) + 1.96 * stats.sem(scores)))
print("Avg CV AUC: " + str(np.average(ROCs)))
print("STDev of CV score: " + str(np.std(ROCs)))
print("StdErr of CV AUC: " + str(stats.sem(ROCs)))
print("95% CI for CV AUC: " + str(np.average(ROCs) - 1.96 * stats.sem(ROCs)) + " to " + str(np.average(ROCs) + 1.96 * stats.sem(ROCs)))

"""
# For LOO and Bootstrapping

# Arek's suggestion to see stdev with 80% of training set used.
#codes_all, _, labels_all, _ = train_test_split(codes_all, labels_all, test_size=0.2, random_state=13)

loo = LeaveOneOut()
kf = KFold(n_splits = 5, shuffle=True, random_state=1395)
predictions = np.zeros(len(labels_all))
confidence = np.zeros(len(labels_all))
for_tsne = np.zeros(len(labels_all))

#for train_index, test_index in kf.split(codes_all):
for train_index, test_index in loo.split(codes_all):   
    X_train, X_test = codes_all[train_index], codes_all[test_index]
    y_train, y_test = labels_all[train_index], labels_all[test_index]
    clf.fit(X_train, y_train)
    predictions[test_index] = clf.predict(X_test)
    confidence[test_index] = abs(clf.decision_function(X_test))
    for_tsne[test_index] = clf.decision_function(X_test)

utility_functions.printListInOrder(predictions)
# if testing human + AI
i = 0
while i < len(names_all):
    if confidence[i] < radio_input_confidence[names_all[i]]:
        predictions[i] = radio_input_classify[names_all[i]]
    i = i + 1   
# end testing human + AI     

fpr, tpr, thresholds = roc_curve(labels_all, predictions)
roc_auc = auc(fpr, tpr)

ROCs = []
for iteration in range(1000):
    indices = resample(range(len(predictions)), random_state=iteration)
    sample_predictions = predictions[indices]
    sample_labels = labels_all[indices]
    fpr, tpr, thresholds = roc_curve(sample_labels, sample_predictions)
    roc_auc_sample = auc(fpr, tpr)
    ROCs.append(roc_auc_sample)
    #print(str(iteration) + " sample AUC: " + str(ROCs[len(ROCs)-1]))


print("Initial AUC: " + str(roc_auc))
print("Avg AUC: " + str(np.average(ROCs)))
print("STDev of CV AUC: " + str(np.std(ROCs)))
print("StdErr of CV AUC: " + str(stats.sem(ROCs)))
# Method 1, direct CI computation
print("95% CI for CV AUC 1: " + str(roc_auc - 1.96 * np.std(ROCs)) + " to " + str(roc_auc + 1.96 * np.std(ROCs)))
# Method 2, implicit CI from the sorted scores 
sorted_scores = np.array(ROCs)
sorted_scores.sort()
confidence_lower = sorted_scores[int(0.025 * len(ROCs))]
confidence_upper = sorted_scores[int(0.975 * len(ROCs))]
print("95% CI for CV AUC 2: " + str(confidence_lower) + " to " + str(confidence_upper))

# Method 3 from https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
# Recommended by Prof. Chen
alpha = 0.95
p = ((1.0-alpha)/2.0)*100
lower = max(0.0, np.percentile(ROCs, p))
p = (alpha+((1.0-alpha)/2.0))*100
upper = min(1.0, np.percentile(ROCs, p))
print("95% CI for CV AUC 3: " + str(lower) + " to " + str(upper))

i = 0
radio_prediction_list = []
while i < len(names_all):
    radio_prediction_list.append(radio_input_classify[names_all[i]])
    i = i + 1
fpr, tpr, thresholds = roc_curve(labels_all, radio_prediction_list)
roc_auc = auc(fpr, tpr)
print("Radiologist AUC: " + str(roc_auc))
#plt.hist(ROCs)
#plt.show()


# Creates a TSNE plot for the deep features generated
for_tsne = for_tsne.reshape(-1, 1)
pca_50 = PCA(n_components=50)
pca_codes = pca_50.fit_transform(codes_all)
scaler = MinMaxScaler()
pca_codes = scaler.fit_transform(pca_codes)
final_values = []
for i in range(len(for_tsne)):
   final_values.append([for_tsne[i][0]])
   for item in pca_codes[i]:
       final_values[i].append(item)
tsne_embedding = TSNE(n_components=2, perplexity=30, init='random', learning_rate=200, n_iter=10000, random_state=0).fit_transform(final_values)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(tsne_embedding[0:len(codes_normal),0], tsne_embedding[0:len(codes_normal),1], edgecolors='none', c="blue", label="normal")
ax.scatter(tsne_embedding[len(codes_normal):,0], tsne_embedding[len(codes_normal):,1], edgecolors='none', c="red", label="cancer")
plt.legend(loc='lower right', fontsize='x-large')
plt.title("t-sne embedding")
plt.xlim([min(tsne_embedding[:,0]-1) - 0.1 *(max(tsne_embedding[:,0]) - min(tsne_embedding[:,0])), max(tsne_embedding[:,0]) + 0.1 *(max(tsne_embedding[:,0]) - min(tsne_embedding[:,0]))])
plt.ylim([min(tsne_embedding[:,1]-1) - 0.1 *(max(tsne_embedding[:,1]) - min(tsne_embedding[:,1])), max(tsne_embedding[:,1]) + 0.1 *(max(tsne_embedding[:,1]) - min(tsne_embedding[:,1]))])
plt.show()

"""
with open('../Results/ROCsNoCropSameDir.txt', 'w') as f:
    for item in ROCs:
        f.write("%s\n" % item)
with open('../Results/predictionsNoCropSameDir.txt', 'w') as f:
    for item in predictions:
        f.write("%s\n" % item)
with open('../Results/confidenceNoCropSameDir.txt', 'w') as f:
    for item in confidence:
        f.write("%s\n" % item)
with open('../Results/labels.txt', 'w') as f:
    for item in labels_all:
        f.write("%s\n" % item)
with open('../Results/names.txt', 'w') as f:
    for item in names_all:
        f.write("%s\n" % item)
            """