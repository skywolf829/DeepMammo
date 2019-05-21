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
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
import tensorflowvgg.vgg19 as vgg19
import utility_functions
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cancer_images = '../Images/CANCER'
contralateral_cancer_images = '../Images/CONTRALATERAL BREAST TO CANCEROUS'
normal_images = '../Images/NORMAL'
codes_path = './codes'
labels_path = './labels'

class_0 = cancer_images
class_1 = normal_images

images, labels = utility_functions.loadImagesFromDir((class_0, class_1), (0,1))
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

np.save(codes_path, codes) 
np.save(labels_path, labels) 

pickle.dump(codes, open('codes', 'wb'))
pickle.dump(labels, open('labels', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.2, random_state=0)

clf = LinearSVC(C=0.0001)
clf.fit(X_train, y_train)

print("Confidence values")
print(clf.decision_function(X_test))

print("Predictions")
print(clf.predict(X_test))

print("Actual values")
print(y_test)

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
