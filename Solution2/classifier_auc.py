import os
import pickle
import argparse
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow_vgg import vgg19

def get_args():
    parser = argparse.ArgumentParser(
        description='Script classifies images in data directory'
    )
    parser.add_argument(
        '-d', '--data', type=str, help='Directory containing image data', required=True
    )
    parser.add_argument(
        '-c', '--codes', type=str, help='Output file path for feature vector', required=True
    )
    parser.add_argument(
        '-l', '--labels', type=str, help='Output file path for labels', required=True
    )
    args = parser.parse_args()
    dataset_folder_path = args.data
    codes_path = args.codes
    labels_path = args.labels
    return dataset_folder_path, codes_path, labels_path

dataset_folder_path, codes_path, labels_path = get_args()

vgg_dir = '../mammogist2/tensorflow_vgg'  # modify
# Make sure vgg exists
if not isdir(vgg_dir):
    raise Exception("VGG directory doesn't exist!")

data_dir = dataset_folder_path
contents = os.listdir(dataset_folder_path)
print(contents)

batch_size = 150
codes_list = []
codes = None
labels = []
batch = []

sess = tf.Session()
print("Session start")
vgg = vgg19.Vgg19()
input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
with tf.name_scope("content_vgg"):
    vgg.build(input_)

files = os.listdir(dataset_folder_path)
print(files)

for i, file in enumerate(files, 1):
    # Add images to the current batch
    print(i)
    file_name = os.path.join(dataset_folder_path, file)
    print(file_name)
    img = load_img(file_name, target_size=(224, 224))
    img_arr = img_to_array(img)
    print("Shape of imgarr:")
    print(img_arr.shape)
    batch.append(img_arr)  # change according to original script
    print(np.shape(batch))

    if file.find("AC") == -1:
        labels.append(0)
    else:
        labels.append(1)

    # Run batch through the network to get codes
    if i % batch_size == 0 or i == len(files):
        # Image batch to pass to VGG network
        images = batch

        # Get the values from the relu6 layer of the VGG network
        feed_dict = {input_: images}
        print("Batch shape:")
        print(np.shape(images))
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

        # Build an array of codes
        if codes is None:
            codes = codes_batch
        else:
            codes = np.append(codes, codes_batch, axis=0)

        # Reset to build next batch
        batch = []
        print('{} images processed'.format(i))

sess.close()

np.save(codes_path, codes)  # user input
print("Shape of codes")
print(np.shape(codes))
np.save(labels_path, labels)  # user input
print("Shape of labels")
print(np.shape(labels))

pickle.dump(codes, open('codes', 'wb'))
pickle.dump(labels, open('labels', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.2, random_state=0)
print(y_train)
print(X_test)

clf = LinearSVC(C=0.0001)
# clf = CalibratedClassifierCV(clf)
clf.fit(X_train, y_train)

print("Confidence values")
print(clf.decision_function(X_test))

# print("Prediction probabilities")
# print(clf.predict_proba(X_test))

print("Predictions")
print(clf.predict(X_test))
print("Actual values")
print(y_test)

# Plot ROC curve
actual = y_test
predictions = clf.decision_function(X_test)
# predictions = clf.predict(X_test)

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
