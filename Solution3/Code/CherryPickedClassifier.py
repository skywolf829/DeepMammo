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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
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


images_normal_train, labels_normal_train, names_normal_train = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/NormalTrain",), (0,))
images_normal_test, labels_normal_test, names_normal_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/NormalTest",), (0,))
images_abnormal_train, labels_abnormal_train, names_abnormal_train = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/AbnormalTrain",), (1,))
images_abnormal_test, labels_abnormal_test, names_abnormal_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/AbnormalTest",), (1,))
images_contralateral_test, labels_contralateral_test, names_contralateral_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInput/ContralateralTest",), (0,))
names_all = np.append(np.append(np.append(names_normal_train, names_normal_test, axis=0), names_abnormal_train, axis=0), names_abnormal_test, axis=0)
labels_all = np.append(np.append(np.append(labels_normal_train, labels_normal_test, axis=0), labels_abnormal_train, axis=0), labels_abnormal_test, axis=0)

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

""" next block is for TSNE plot """
codes_all = np.append(np.append(np.append(codes_normal_train, codes_normal_test, axis=0), codes_cancer_train, axis=0), codes_cancer_test, axis=0)
#codes_all = PCA(n_components=50).fit_transform(codes_all)
tsne_embedding = TSNE(n_components=2, perplexity=5).fit_transform(codes_all)
json_dict = {}
i=0
for name in names_all:
    json_dict[name] = {}
    json_dict[name]["position"] = tsne_embedding[i].tolist()
    json_dict[name]["label"] = str(labels_all[i])
    i = i + 1
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(tsne_embedding[0:len(codes_normal_train)+len(codes_normal_test),0], tsne_embedding[0:len(codes_normal_train)+len(codes_normal_test),1], edgecolors='none', c="blue", label="normal")
ax.scatter(tsne_embedding[len(codes_normal_train)+len(codes_normal_test):,0], tsne_embedding[len(codes_normal_train)+len(codes_normal_test):,1], edgecolors='none', c="red", label="cancer")
plt.legend(loc='lower right', fontsize='x-large')
plt.title("t-sne embedding")
plt.xlim([min(tsne_embedding[:,0]-1), max(tsne_embedding[:,0]+1)])
plt.ylim([min(tsne_embedding[:,1]-1), max(tsne_embedding[:,1]+1)])
plt.show()


clf = LinearSVC(C=0.0001)

X_train = np.append(codes_normal_train, codes_cancer_train, axis=0)
X_test = np.append(codes_normal_test, codes_cancer_test, axis=0)

y_train = np.append(labels_normal_train, labels_abnormal_train, axis=0)
y_test = np.append(labels_normal_test, labels_abnormal_test, axis=0)

names_train = np.append(names_normal_train, names_abnormal_train, axis=0)
names_test = np.append(names_normal_test, names_abnormal_test, axis=0)

C_values = [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
C_value_scores = []
for spot in range(len(C_values)):
    clf = LinearSVC(C=C_values[spot])
    kFolds = 5
    iterations = 1000
    random_state = 4597834
    i = 0
    averageScore = 0
    rollingAverage = 0
    rkf = RepeatedKFold(n_splits=kFolds, n_repeats=iterations, random_state=random_state)

    for train_index, test_index in rkf.split(X_train):
        X_train_CV, X_test_CV = X_train[train_index], X_train[test_index]
        y_train_CV, y_test_CV = y_train[train_index], y_train[test_index]
        clf.fit(X_train_CV, y_train_CV)
        score = clf.score(X_test_CV, y_test_CV)
        averageScore = averageScore + score
        rollingAverage = rollingAverage + score
        i = i + 1
        if i % kFolds == 0:
            print("Average for " + str(kFolds) + "-split " + str(i / kFolds) + ": " + str (rollingAverage / kFolds))
            rollingAverage = 0

    averageScore = averageScore / i

    print("Average score: " + str(averageScore))
    C_value_scores.append(averageScore)
print(C_value_scores)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Final score: " + str(score))
print("Overall score: " + str(clf.score(np.append(X_train, X_test, axis=0), np.append(y_train, y_test, axis=0))))


model_confidence = {}
model_classification = {}
model_classification_contralateral = {}
model_confidence_contralateral = {}

confidence_values = clf.decision_function(X_test)
scaler = MinMaxScaler(feature_range=(-1, 1))
confidence_values = scaler.fit_transform(np.array(confidence_values).reshape(-1, 1)).reshape(-1)
i = 0
for item in confidence_values:
    model_confidence[names_test[i]] = abs(item)
    i = i + 1

predictions = clf.predict(X_test)
i = 0
for item in predictions:
    model_classification[names_test[i]] = item
    i = i + 1

predictions_contralateral = clf.predict(codes_contralateral)
i = 0
for item in predictions_contralateral:
    model_classification_contralateral[names_contralateral_test[i]] = item
    i = i + 1

confidence_values_contralateral = clf.decision_function(codes_contralateral)
i = 0
for item in confidence_values_contralateral:
    model_confidence_contralateral[names_contralateral_test[i]] = abs(item)
    i = i + 1

model_confidence_all = {}
model_classification_all = {}

confidence_values_all = clf.decision_function(np.append(X_train, X_test, axis=0))
i = 0
for item in confidence_values_all:
    model_confidence_all[names_all[i]] = abs(item)
    i = i + 1

predictions_all = clf.predict(np.append(X_train, X_test, axis=0))
i = 0
for item in predictions_all:
    model_classification_all[names_all[i]] = item
    i = i + 1

for name in names_all:
    if name in model_confidence_all.keys():
        json_dict[name]["model_confidence"] = str(model_confidence_all[name])
    if name in model_classification_all.keys():
        json_dict[name]["model_classification"] = str(model_classification_all[name])
    if name in radio_input_classify.keys():
        json_dict[name]["radiologist_classification"] = str(radio_input_classify[name])
    else:
        json_dict[name]["radiologist_classification"] = "N/A"
    if name in radio_input_confidence.keys():
        json_dict[name]["radiologist_confidence"] = str(radio_input_confidence[name])
    else:
        json_dict[name]["radiologist_confidence"] = "N/A"
    i = i + 1
with open('js/VisualizationInformation.txt', 'w') as json_file:
    json.dump(json_dict, json_file)

#utility_functions.printListInOrder(y_test)
#print("break")
#utility_functions.printDictionaryInOrder(names_test, radio_input_classify)
#print("break")
#utility_functions.printDictionaryInOrder(names_test, radio_input_confidence)


radio_confidence = []
for name in names_test:
    radio_confidence.append(radio_input_classify[name])
fpr, tpr, thresholds = roc_curve(y_test, radio_confidence)
roc_auc = auc(fpr, tpr)
print("AUC: " + str(roc_auc))

confidence_values_model = []
confidence_values_radiologist = []
for i in range(len(names_test)):
    if names_test[i] in radio_input_confidence.keys():
        if model_classification[names_test[i]] == 1:
            confidence_values_model.append(-model_confidence[names_test[i]])
        else:    
            confidence_values_model.append(model_confidence[names_test[i]])
        if radio_input_confidence[names_test[i]] == 1:
            confidence_values_radiologist.append(-radio_input_confidence[names_test[i]])
        else:
            confidence_values_radiologist.append(radio_input_confidence[names_test[i]])
scaler = MinMaxScaler(feature_range=(0, 1))
confidence_values_model = scaler.fit_transform(np.array(confidence_values_model).reshape(-1, 1)).reshape(-1)
r, p = scipy.stats.pearsonr(confidence_values_model, confidence_values_radiologist)
print("Pearson r: " + str(r) + ", p-value: " + str(p))

plt.plot(fpr, tpr, 'darkorange',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right', fontsize='x-large')
plt.title("ROC Curve - Linear SVM")
plt.plot([0, 1], [0, 1], color='#67809f', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()


"""

The following code is to add a voting system in hopes to increase accuracy

"""
for i in range(len(names_test)):
    name = names_test[i]
    radio_score = radio_input_confidence[name]
    model_score = model_confidence[name]
    if radio_score > model_score:
        predictions[i] = radio_input_classify[name]
        confidence_values[i] = radio_score
    else:
        predictions[i] = model_classification[name]
        confidence_values[i] = model_score
    if predictions[i] == 0:
        confidence_values[i] = -confidence_values[i]

numCorrect = 0
for i in range(len(names_test)):
    if predictions[i] == y_test[i]:
        numCorrect = numCorrect + 1
newAccuracy = float(numCorrect) / len(names_test)
print("Human input accuracy: " + str(newAccuracy))
fpr, tpr, thresholds = roc_curve(y_test, confidence_values)
roc_auc = auc(fpr, tpr)
print("New AUC: "+str(roc_auc))
plt.plot(fpr, tpr, 'darkorange',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right', fontsize='x-large')
plt.title("ROC Curve - Voting System")
plt.plot([0, 1], [0, 1], color='#67809f', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()
#utility_functions.printListInOrder(confidence_values)
#print("break")
#utility_functions.printListInOrder(predictions)

