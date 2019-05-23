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

cancer_images = '../Images/CANCER'
contralateral_cancer_images = '../Images/CONTRALATERAL BREAST TO CANCEROUS'
normal_images = '../Images/NORMAL'
normal_images_bilateral = "../Images/Bilateral/Normal"
cancer_images_bilateral = "../Images/Bilateral/Cancer"
codes_path = './codes'
labels_path = './labels'
names_path = './names'

class_0 = normal_images_bilateral
class_1 = cancer_images_bilateral

images, labels, names = utility_functions.loadImagesFromDir((class_0, class_1), (0,1))
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

codes = np.array(codes)
labels = np.array(labels)
names = np.array(names)

clf = LinearSVC(C=0.0001)
#clf = SVC(kernel='linear', gamma='auto', C=.0001)
#clf = SVC(kernel='rbf', gamma='scale', C=1)
#clf = KNeighborsClassifier(3)
#clf = DecisionTreeClassifier(max_depth=5, max_features=1)
#clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#clf = MLPClassifier(alpha=1, max_iter=1000)
#clf = AdaBoostClassifier()
#clf = GaussianNB()

"""
kFolds = 5
iterations = 1000
random_state = 4597834
i = 0
averageScore = 0
rollingAverage = 0
rkf = RepeatedKFold(n_splits=kFolds, n_repeats=iterations, random_state=random_state)

for train_index, test_index in rkf.split(codes):
    X_train, X_test = codes[train_index], codes[test_index]
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
"""
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(codes, labels, names, test_size=0.3, random_state=261963)
names_testing_dict = {}
for item in names:
    if item in names_test:
        names_testing_dict[item] = 1
    else:
        names_testing_dict[item] = ""

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Final score: " + str(score))
confidence_values = clf.decision_function(codes)
predictions = clf.predict(codes)

model_confidence = {}
model_classification = {}
i = 0
for item in confidence_values:
    model_confidence[names[i]] = abs(item)
    i = i + 1
i = 0
for item in predictions:
    model_classification[names[i]] = item
    i = i + 1

radio_input_classify, radio_input_confidence = utility_functions.loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)
pickle.dump(radio_input_classify, open('radio_input_classify', 'wb'))
pickle.dump(radio_input_confidence, open('radio_input_confidence', 'wb'))
pickle.dump(model_classification, open('model_input_classify', 'wb'))
pickle.dump(model_confidence, open('model_input_confidence', 'wb'))

pickle.dump(codes, open('codes', 'wb'))
pickle.dump(labels, open('labels', 'wb'))
pickle.dump(names, open('names', 'wb'))

utility_functions.printDictionaryInOrder(names, names_testing_dict)
print("break")
utility_functions.printDictionaryInOrder(names, radio_input_classify)
print("break")
utility_functions.printDictionaryInOrder(names, radio_input_confidence)
#print("Confidence values")
#print(clf.decision_function(X_test))

#print("Predictions")
#print(clf.predict(X_test))

#print("Actual values")
#print(y_test)

#print("Accuracy score")
#print(clf.score(X_test, y_test))

# Plot ROC curve
actual = y_test
predictions = clf.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(actual, predictions)
roc_auc = auc(fpr, tpr)
print("AUC")
print(roc_auc)

plt.plot(fpr, tpr, 'darkorange',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right', fontsize='x-large')
plt.plot([0, 1], [0, 1], color='#67809f', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.show()
