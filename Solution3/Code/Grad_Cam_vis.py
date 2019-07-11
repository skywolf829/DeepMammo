import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import pickle
import cv2
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
import GuideReLU as GReLU
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
import tensorflowvgg.vgg19SVM as vgg19
import utility_functions
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
from skimage.transform import resize
import json

def guided_BP(image, label_id, last_layer_weights):	
	g = tf.get_default_graph()
	with g.gradient_override_map({'Relu': 'GuidedRelu'}):
		label_vector = tf.placeholder("float", [None, 2])
		input_image = tf.placeholder("float", [None, 224, 224, 3])

		vgg = vgg19.Vgg19SVM(last_layer_weights=last_layer_weights)
		with tf.name_scope("content_vgg"):
		    vgg.build(input_image)

		cost = vgg.fc7*label_vector
	
		# Guided backpropagtion back to input layer
		gb_grad = tf.gradients(cost, input_image)[0]

		init = tf.global_variables_initializer()
	
	# Run tensorflow 
	with tf.Session(graph=g) as sess:    
		sess.run(init)
		output = [0.0]*vgg.prob.get_shape().as_list()[1] #one-hot embedding for desired class activations
		if label_id == -1:
			prob = sess.run(vgg.prob, feed_dict={input_image:image})

			#creating the output vector for the respective class
			index = np.argmax(prob)
			print("Predicted_class: " + str(index))
			output[index] = 1.0

		else:
			output[label_id] = 1.0
		output = np.array(output)
		gb_grad_value = sess.run(gb_grad, feed_dict={input_image:image, label_vector: output.reshape((1,-1))})

	return gb_grad_value[0] 


def visualize(img, cam, filename,gb_viz):
    img = img / 255.0
    gb_viz = np.dstack((
            gb_viz[:, :, 2],
            gb_viz[:, :, 1],
            gb_viz[:, :, 0],
        ))

    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()
  
    fig, ax = plt.subplots(nrows=1,ncols=3)

    plt.subplot(141)
    plt.axis("off")
    imgplot = plt.imshow(img)

    plt.subplot(142)
    gd_img = gb_viz*np.minimum(0.25,cam).reshape(224,224,1)
    x = gd_img
    x = np.squeeze(x)
    
    #normalize tensor
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
   
    x = np.clip(x, 0, 255).astype('uint8')

    plt.axis("off")
    imgplot = plt.imshow(x, vmin = 0, vmax = 20)

    cam = (cam*-1.0) + 1.0
    cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET))
    plt.subplot(143)
    plt.axis("off")

    imgplot = plt.imshow(cam_heatmap)

    plt.subplot(144)
    plt.axis("off")
    
    cam_heatmap = cam_heatmap/255.0

    fin = (img*0.7) + (cam_heatmap*0.3)
    imgplot = plt.imshow(fin)

    plt.savefig("../Images/GradCamVisualizations/NoCrop/"+filename, dpi=600)
    plt.close(fig)

def run_viz(image_name, label_to_visualize):
    test_index = [np.where(names_all==im_name)[0][0]]
    train_index = []
    for i in range(len(names_all)):
        if i is not test_index[0]:
            train_index.append(i)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.variable_scope("ForSVM"):
            print("Session start")
            vgg = vgg19.Vgg19SVM()
            input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
            vgg.build(input_)
            tf.get_default_graph().finalize()
            # Get the values from the relu6 layer of the VGG network
            feed_dict_all = {input_: images_all}
            feed_dict_normal = {input_: images_normal}
            feed_dict_cancer = {input_: images_cancer}
            codes_normal = sess.run(vgg.relu6, feed_dict=feed_dict_normal)
            codes_cancer = sess.run(vgg.relu6, feed_dict=feed_dict_cancer)        
            codes_all = np.append(codes_normal, codes_cancer, axis=0)
    tf.reset_default_graph()

    clf = LinearSVC(C=0.0001)
    X_train, X_test = codes_all[train_index], codes_all[test_index]
    y_train, y_test = labels_all[train_index], labels_all[test_index]
    clf.fit(X_train, y_train)

    label_id = label_to_visualize
    print("Predicted: " + str(clf.predict(X_test)[0]))
    weights = clf.coef_[0]
    biases = clf.intercept_
    print("Manual computation: " + str(np.dot(X_test[0], weights)+biases[0]))
    print("Decision function:  " + str(clf.decision_function(X_test)[0]))
    w = []
    for i in weights:
        w.append([np.float32(-i), np.float32(i)])
    w = np.array(w)

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        with tf.name_scope("content_vgg"):
            vgg = vgg19.Vgg19SVM(last_layer_weights=[w, np.float32([-biases[0], biases[0]])])
        #define your tensor placeholders for, labels and images
        label_vector = tf.placeholder("float", [None, 2])
        input_image = tf.placeholder("float", [1, 224, 224, 3])
        label_index = tf.placeholder("int64", ())

        vgg.build(input_image)
        """
        tf.get_default_graph().finalize()
        feed_dict = {input_image: images_all[test_index]}
        codes = sess.run(vgg.prob, feed_dict=feed_dict)
        """
        print("Built vgg")
        #get the output neuron corresponding to the class of interest (label_id)
        cost = vgg.fc7*label_vector

        # Get last convolutional layer gradients for generating gradCAM++ visualization
        target_conv_layer = vgg.conv5_4
        target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]
        
        #first_derivative
        first_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad 	
        
        #second_derivative
        second_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad 

        #triple_derivative
        triple_derivative = tf.exp(cost)[0][label_index]*target_conv_layer_grad*target_conv_layer_grad*target_conv_layer_grad  

        img1 = images_all[test_index][0]
        output = [0.0]*vgg.prob.get_shape().as_list()[1] #one-hot embedding for desired class activations
            #creating the output vector for the respective class
        print("Pre-sess.run")
        if label_id == -1:
            prob_val = sess.run(vgg.prob, feed_dict={input_image:[img1]})
            #creating the output vector for the respective class
            index = np.argmax(prob_val)
            orig_score = prob_val[0][index]
            print("Predicted_class: " + str(index))
            output[index] = 1.0
            label_id = index
        else:
            output[label_id] = 1.0	
        output = np.array(output)
        print("label: " + str(label_id))
        #conv_output = sess.run(target_conv_layer, feed_dict={input_image:[img1], label_index:label_id, label_vector: output.reshape((1,-1))})
        conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run([target_conv_layer, first_derivative, second_derivative, triple_derivative], feed_dict={input_image:[img1], label_index:label_id, label_vector: output.reshape((1,-1))})
        print("Another big sess.run")
        global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)
        print("normalizing the alphas")
        """	
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
        
        alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
        """

        alphas_thresholding = np.where(weights, alphas, 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))


        alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))


        
        deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
        #print deep_linearization_weights
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0   

        cam = resize(cam, (224,224))
        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0    
        cam = resize(cam, (224,224))
        print("Starting guided backprop")
        gb = guided_BP([img1], label_id, [w, np.float32([-biases[0], biases[0]])])
        print("Visualizing")
        visualize(img1, cam, im_name.split(".")[0]+"-gradcam_viz"+str(label_to_visualize)+".png", gb) 
        print("Done")



radio_input_classify, radio_input_confidence = utility_functions.loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)

images_normal, labels_normal, names_normal = utility_functions.loadImagesFromDir(("../Images/Normal",), (0,))
images_cancer, labels_cancer, names_cancer = utility_functions.loadImagesFromDir(("../Images/Cancer",), (1,))
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

for im_name in names_all:
    run_viz(im_name, 0)
    run_viz(im_name, 1)