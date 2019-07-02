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
import utils
import json
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

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
codes_train, codes_test, labels_train, labels_test = train_test_split(codes_all, labels_all, test_size=0.2, random_state=13)
clf.fit(codes_train, labels_train)
predictions = clf.predict(codes_test)

eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': "GuidedRelu'"}):
        images = tf.placeholder("float", [1, 244, 244, 3])
        labels = tf.placeholder(tf.float32, [1, 1])
        vgg = vgg19.Vgg19()
        vgg.build(images)
        cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(vgg.prob)), axis=1)
        print('cost:', cost)
        # cost = tf.reduce_sum((vgg.prob - labels) ** 2)
        
        
        # gradient for partial linearization. We only care about target visualization class. 
        y_c = tf.reduce_sum(tf.multiply(vgg.fc8, labels), axis=1)
        print('y_c:', y_c)
        # Get last convolutional layer gradient for generating gradCAM visualization
        target_conv_layer = vgg.pool5
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, images)[0]

        init = tf.global_variables_initializer()

with tf.Session(graph=eval_graph) as sess:    
    sess.run(init)
    
    prob = sess.run(vgg.prob, feed_dict={images: batch_img})
    
    gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: batch_label})
    
    
    for i in range(1):
        # VGG16 use BGR internally, so we manually change BGR to RGB
        gradBGR = gb_grad_value[i]
        gradRGB = np.dstack((
            gradBGR[:, :, 2],
            gradBGR[:, :, 1],
            gradBGR[:, :, 0],
        ))
        utils.visualize(batch_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], gradRGB)