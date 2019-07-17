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
import lightgbm as lgb

codes_path = './codes'
labels_path = './labels'
names_path = './names'
radio_input_classify, radio_input_confidence = utility_functions.loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)


images_normal, labels_normal, names_normal = utility_functions.loadImagesFromDir(("../Images/CroppedOriginalRotation/Normal",), (0,))
images_cancer, labels_cancer, names_cancer = utility_functions.loadImagesFromDir(("../Images/CroppedOriginalRotation/Cancer",), (1,))

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

rotate90 = utility_functions.rotateImages(images_all, 90, False, False)
rotate180 = utility_functions.rotateImages(images_all, 180, False, False)
rotate270 = utility_functions.rotateImages(images_all, 270, False, False)

mirrored = utility_functions.rotateImages(images_all, None, False, True)
rotate90mirrored = utility_functions.rotateImages(rotate90, None, False, True)
rotate180mirrored = utility_functions.rotateImages(rotate180, None, False, True)
rotate270mirrored = utility_functions.rotateImages(rotate270, None, False, True)

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
feed_dict_90 = {input_: rotate90}
feed_dict_180 = {input_: rotate180}
feed_dict_270 = {input_: rotate270}
feed_dict_mirrored = {input_: mirrored}
feed_dict_90mirrored = {input_: rotate90mirrored}
feed_dict_180mirrored = {input_: rotate180mirrored}
feed_dict_270mirrored = {input_: rotate270mirrored}

codes_normal = sess.run(vgg.relu6, feed_dict=feed_dict_normal)
codes_cancer = sess.run(vgg.relu6, feed_dict=feed_dict_cancer)
codes_all = np.append(codes_normal, codes_cancer, axis=0)
codes_all_90 = sess.run(vgg.relu6, feed_dict=feed_dict_90)
codes_all_180 = sess.run(vgg.relu6, feed_dict=feed_dict_180)
codes_all_270 = sess.run(vgg.relu6, feed_dict=feed_dict_270)
codes_all_mirrored = sess.run(vgg.relu6, feed_dict=feed_dict_mirrored)
codes_all_90mirrored = sess.run(vgg.relu6, feed_dict=feed_dict_90mirrored)
codes_all_180mirrored = sess.run(vgg.relu6, feed_dict=feed_dict_180mirrored)
codes_all_270mirrored = sess.run(vgg.relu6, feed_dict=feed_dict_270mirrored)
sess.close()

final_feature_fectors = np.concatenate((codes_all, codes_all_90, codes_all_180, codes_all_270, codes_all_mirrored, codes_all_90mirrored, codes_all_180mirrored, codes_all_270mirrored), axis=1)
#labels_all = np.concatenate((labels_all, labels_all, labels_all, labels_all, labels_all, labels_all, labels_all, labels_all), axis=0)
pca = PCA(n_components=50).fit(final_feature_fectors)
clf = LinearSVC(C=0.001, max_iter=1000000)

# For LOO and Bootstrapping

# Arek's suggestion to see stdev with 80% of training set used.
#codes_all, _, labels_all, _ = train_test_split(codes_all, labels_all, test_size=0.2, random_state=13)

loo = LeaveOneOut()
predictions = np.zeros(len(labels_all))
confidence = np.zeros(len(labels_all))
for_tsne = np.zeros(len(labels_all))
conf_roc = np.zeros(len(labels_all))
for train_index, test_index in loo.split(codes_all):   
    X_train = np.concatenate((codes_all[train_index], codes_all_90[train_index], codes_all_180[train_index], codes_all_270[train_index], codes_all_mirrored[train_index], codes_all_90mirrored[train_index], codes_all_180mirrored[train_index], codes_all_270mirrored[train_index]), axis=1)
    X_train = pca.transform(X_train)
    #X_test = codes_all[test_index]
    X_test = np.concatenate((codes_all[test_index], codes_all_90[test_index], codes_all_180[test_index], codes_all_270[test_index], codes_all_mirrored[test_index], codes_all_90mirrored[test_index], codes_all_180mirrored[test_index], codes_all_270mirrored[test_index]), axis=1)
    X_test = pca.transform(X_test)
    #y_train = np.concatenate((labels_all[train_index],labels_all[train_index], labels_all[train_index], labels_all[train_index], labels_all[train_index], labels_all[train_index], labels_all[train_index], labels_all[train_index]), axis=0)
    y_train = labels_all[train_index]
    y_test = labels_all[test_index]
    clf.fit(X_train, y_train)
    predictions[test_index] = clf.predict(X_test)
    confidence[test_index] = abs(clf.decision_function(X_test))
    conf_roc[test_index] = clf.decision_function(X_test)
    for_tsne[test_index] = clf.decision_function(X_test)
    print("Finished LOO split " + str(test_index))
    print(X_train.shape)
    print(y_train.shape)

utility_functions.printListInOrder(predictions)
#utility_functions.printListInOrder(confidence)
tn, fp, fn, tp = confusion_matrix(labels_all, predictions).ravel()
acc = accuracy_score(labels_all, predictions)
fpr, tpr, thresholds = roc_curve(labels_all, conf_roc)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'darkorange',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right', fontsize='x-large')
plt.title("ROC Curve: Machine")
plt.plot([0, 1], [0, 1], color='#67809f', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()
print("Machine accuracy: " + str(acc))
print("Machine FPR: " + str(fp/len(labels_normal)))
print("Machine TPR: " + str(tp/len(labels_cancer)))
print("Machine AUC: " + str(roc_auc))

#utility_functions.printListInOrder(predictions)
# if testing human + AI
i = 0
while i < len(names_all):
    if confidence[i] < radio_input_confidence[names_all[i]]:
        predictions[i] = radio_input_classify[names_all[i]]
        conf_roc[i] = radio_input_classify[names_all[i]]
        if(predictions[i] == 0):
            conf_roc[i] = -conf_roc[i]
    i = i + 1   

#utility_functions.printListInOrder(predictions)
# end testing human + AI     
fpr, tpr, thresholds = roc_curve(labels_all, conf_roc)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'darkorange',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right', fontsize='x-large')
plt.title("ROC Curve: Machine + Radiologist")
plt.plot([0, 1], [0, 1], color='#67809f', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()

ROCs = []
for iteration in range(1000):
    indices = resample(range(len(predictions)), random_state=iteration)
    sample_predictions = predictions[indices]
    sample_scores = conf_roc[indices]
    sample_labels = labels_all[indices]
    fpr, tpr, thresholds = roc_curve(sample_labels, sample_scores)
    roc_auc_sample = auc(fpr, tpr)
    ROCs.append(roc_auc_sample)
    #print(str(iteration) + " sample AUC: " + str(ROCs[len(ROCs)-1]))

tn, fp, fn, tp = confusion_matrix(labels_all, predictions).ravel()
acc = accuracy_score(labels_all, predictions)

print("Machine+radiologist accuracy: " + str(acc))
print("Machine+radiologist FPR: " + str(fp/len(labels_normal)))
print("Machine+radiologist TPR: " + str(tp/len(labels_cancer)))
print("Machine+radiologist AUC: " + str(roc_auc))
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
radio_conf_list = []
while i < len(names_all):
    radio_prediction_list.append(radio_input_classify[names_all[i]])
    radio_conf_list.append(radio_input_confidence[names_all[i]])
    if(radio_prediction_list[i] == 0):
        radio_conf_list[i] = -radio_conf_list[i]
    i = i + 1
fpr, tpr, thresholds = roc_curve(labels_all, radio_prediction_list)
roc_auc = auc(fpr, tpr)
tn, fp, fn, tp = confusion_matrix(labels_all, radio_prediction_list).ravel()
acc = accuracy_score(labels_all, radio_prediction_list)
print("Radiologist accuracy: " + str(acc))
print("Radiologist FPR: " + str(fp/len(labels_normal)))
print("Radiologist TPR: " + str(tp/len(labels_cancer)))
print("Radiologist AUC: " + str(roc_auc))
#plt.hist(ROCs)
#plt.show()


# Creates a TSNE plot for the deep features generated
for_tsne = for_tsne.reshape(-1, 1)
pca_50 = PCA(n_components=50)
pca_codes = pca_50.fit_transform(final_feature_fectors)
scaler = MinMaxScaler()
pca_codes = scaler.fit_transform(pca_codes)
final_values = []
for i in range(len(for_tsne)):
   final_values.append([for_tsne[i][0]])
   for item in pca_codes[i]:
       final_values[i].append(item)
tsne_embedding = TSNE(n_components=2, perplexity=60, init='random', learning_rate=200, n_iter=10000, random_state=0).fit_transform(final_values)
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
with open('../Results/ROCsNoCropAllRotations.txt', 'w') as f:
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