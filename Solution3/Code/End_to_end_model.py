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
import lightgbm as lgbm
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
import types


def new_forward_vgg19(self, x):
    # conv1
    print("input " + str(x.size()))
    #x1 = self._modules["avgpool"](x)
    #print("x1 " + str(x1.size()))
    x1 = self._modules["_features"](x)
    print("x1 " + str(x1.size()))
    x1 = torch.Tensor.flatten(x1, 1, -1)
    print("x1 " + str(x1.size()))
    x2 = self._modules["linear0"](x1)
    print("x2 " + str(x2.size()))
    x3 = self._modules["relu0"](x2)
    print("x3 " + str(x3.size()))
    return x3

model_name = "inceptionv4"
classifier_name = "LGBM"
preprocessingType = 2
use_radiologist_gist = True

use_rotations = False
use_PCA = False
show_charts = False
print_statements_debug = False
save_AUC_chart = False
print_TPRandFPR = False
save_results_and_do_post_hoc_tests = False
do_TSNE = False


if(model_name == "vgg19"):
    model = pretrainedmodels.vgg19(num_classes=1000, pretrained='imagenet')
    model.eval()
    model.forward = types.MethodType(new_forward_vgg19, model)
elif(model_name == "inceptionv4"):
    model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')
    model.eval()

for param in model.parameters():
    param.requires_grad = False
model.cuda(device="cuda")

codes_path = './codes'
labels_path = './labels'
names_path = './names'
radio_input_classify, radio_input_confidence = utility_functions.loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)


images_normal, labels_normal, names_normal = utility_functions.loadImagesFromDirTorch(("../Images/preprocessing"+str(preprocessingType)+"/normal",), (0,), model)
images_cancer, labels_cancer, names_cancer = utility_functions.loadImagesFromDirTorch(("../Images/preprocessing"+str(preprocessingType)+"/cancer",), (1,), model)

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

images_all = []
for img in images_normal:
    images_all.append(img)
for img in images_cancer:
    images_all.append(img)
images_all = np.array(images_all)

rotate90 = utility_functions.rotateImages(images_all, 90, False, False, swapaxes=True, numpy=True)
rotate180 = utility_functions.rotateImages(images_all, 180, False, False, swapaxes=True, numpy=True)
rotate270 = utility_functions.rotateImages(images_all, 270, False, False, swapaxes=True, numpy=True)

mirrored = utility_functions.rotateImages(images_all, None, False, True, swapaxes=True, numpy=True)
rotate90mirrored = utility_functions.rotateImages(rotate90, None, False, True, swapaxes=True, numpy=True)
rotate180mirrored = utility_functions.rotateImages(rotate180, None, False, True, swapaxes=True, numpy=True)
rotate270mirrored = utility_functions.rotateImages(rotate270, None, False, True, swapaxes=True, numpy=True)

images_all = torch.from_numpy(images_all)
rotate90 = torch.from_numpy(rotate90)
rotate180 = torch.from_numpy(rotate180)
rotate270 = torch.from_numpy(rotate270)
mirrored = torch.from_numpy(mirrored)
rotate90mirrored = torch.from_numpy(rotate90mirrored)
rotate180mirrored = torch.from_numpy(rotate180mirrored)
rotate270mirrored = torch.from_numpy(rotate270mirrored)

images_all = images_all.cuda()
rotate90 = rotate90.cuda()
rotate180 = rotate180.cuda()
rotate270 = rotate270.cuda()
mirrored = mirrored.cuda()
rotate90mirrored = rotate90mirrored.cuda()
rotate180mirrored = rotate180mirrored.cuda()
rotate270mirrored = rotate270mirrored.cuda()

if(model_name == "vgg19"):
    codes_all = model(images_all)
    codes_rotate90 = model(rotate90)
    codes_rotate180 = model(rotate180)
    codes_rotate270 = model(rotate270)
    codes_mirrored = model(mirrored)
    codes_rotate90mirrored = model(rotate90mirrored)
    codes_rotate180mirrored = model(rotate180mirrored)
    codes_rotate270mirrored = model(rotate270mirrored)

elif(model_name == "inceptionv4"):
    codes_all = model.features(images_all)
    codes_all = model._modules["avg_pool"](codes_all).flatten(1, -1)

    codes_rotate90 = model.features(rotate90)
    codes_rotate90 = model._modules["avg_pool"](codes_rotate90).flatten(1, -1)

    codes_rotate180 = model.features(rotate180)
    codes_rotate180 = model._modules["avg_pool"](codes_rotate180).flatten(1, -1)

    codes_rotate270 = model.features(rotate270)
    codes_rotate270 = model._modules["avg_pool"](codes_rotate270).flatten(1, -1)

    codes_mirrored = model.features(mirrored)
    codes_mirrored = model._modules["avg_pool"](codes_mirrored).flatten(1, -1)

    codes_rotate90mirrored = model.features(rotate90mirrored)
    codes_rotate90mirrored = model._modules["avg_pool"](codes_rotate90mirrored).flatten(1, -1)

    codes_rotate180mirrored = model.features(rotate180mirrored)
    codes_rotate180mirrored = model._modules["avg_pool"](codes_rotate180mirrored).flatten(1, -1)

    codes_rotate270mirrored = model.features(rotate270mirrored)
    codes_rotate270mirrored = model._modules["avg_pool"](codes_rotate270mirrored).flatten(1, -1)

codes_all = codes_all.cpu()
codes_all = codes_all.detach().numpy()
codes_rotate90 = codes_rotate90.cpu()
codes_rotate90 = codes_rotate90.detach().numpy()
codes_rotate180 = codes_rotate180.cpu()
codes_rotate180 = codes_rotate180.detach().numpy()
codes_rotate270 = codes_rotate270.cpu()
codes_rotate270 = codes_rotate270.detach().numpy()
codes_mirrored = codes_mirrored.cpu()
codes_mirrored = codes_mirrored.detach().numpy()
codes_rotate90mirrored = codes_rotate90mirrored.cpu()
codes_rotate90mirrored = codes_rotate90mirrored.detach().numpy()
codes_rotate180mirrored = codes_rotate180mirrored.cpu()
codes_rotate180mirrored = codes_rotate180mirrored.detach().numpy()
codes_rotate270mirrored = codes_rotate270mirrored.cpu()
codes_rotate270mirrored = codes_rotate270mirrored.detach().numpy()

if(use_rotations):
    codes_all = np.concatenate((codes_all, codes_rotate90, codes_rotate180, codes_rotate270, codes_mirrored, codes_rotate90mirrored, codes_rotate180mirrored, codes_rotate270mirrored), axis=1)

if(use_radiologist_gist):
    codes_all = codes_all.tolist()
    for i in range(len(names_all)):
        codes_all[i] = np.concatenate((codes_all[i], [1*100] if(radio_input_classify[names_all[i]] == 1) else [-1*100]), axis=None)
        codes_all[i] = np.concatenate((codes_all[i], [radio_input_confidence[names_all[i]]*100]), axis=None)
    codes_all = np.array(codes_all)


pca = PCA(n_components=len(codes_all)).fit(codes_all)
if(use_PCA):
    codes_all = pca.transform(codes_all)
print("Codes shape: " + str(codes_all.shape))

codes_normal = codes_all[0:len(names_normal)]
codes_cancer = codes_all[len(names_normal):len(codes_all)]


clf = LinearSVC(C=1000, max_iter=10000)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 1
params['max_depth'] = 1000

# For LOO and Bootstrapping
loo = LeaveOneOut()
predictions = np.zeros(len(labels_all))
confidence = np.zeros(len(labels_all))
for_tsne = np.zeros(len(labels_all))
conf_roc = np.zeros(len(labels_all))
for train_index, test_index in loo.split(codes_all):   
    X_train, X_test = codes_all[train_index], codes_all[test_index]
    y_train, y_test = labels_all[train_index], labels_all[test_index]
    if(classifier_name == "SVM"):
        clf.fit(X_train, y_train)
    elif(classifier_name == "LGBM"):
        clf = lgbm.train(params, lgbm.Dataset(X_train, y_train), 100)
    predictions[test_index] = 1 if clf.predict(X_test) > 0.5 else 0
    if(classifier_name == "SVM"):
        confidence[test_index] = abs(clf.decision_function(X_test))
        conf_roc[test_index] = clf.decision_function(X_test)
        for_tsne[test_index] = clf.decision_function(X_test)
    elif(classifier_name == "LGBM"):
        confidence[test_index] = clf.predict(X_test)
        conf_roc[test_index] = clf.predict(X_test)
        for_tsne[test_index] = clf.predict(X_test)
    if(print_statements_debug):
        print(str(predictions[test_index]) + " " + str(labels_all[test_index]))

title = "ROC Curve - " + model_name + " features into " + classifier_name
if(use_radiologist_gist):
    title = title + " with radiologist gist"
customfont = {'fontname':'Helvetica'}
tn, fp, fn, tp = confusion_matrix(labels_all, predictions).ravel()
acc = accuracy_score(labels_all, predictions)
fpr, tpr, thresholds = roc_curve(labels_all, conf_roc)
if(print_TPRandFPR):
    print("FPR:")
    utility_functions.printListInOrder(fpr)
    print("TPR:")
    utility_functions.printListInOrder(tpr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'darkorange',
         label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right', fontsize='x-large')
plt.title(title, **customfont)
plt.plot([0, 1], [0, 1], color='#67809f', linestyle='--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate', fontsize=14, **customfont)
plt.xlabel('False Positive Rate', fontsize=14, **customfont)
if(show_charts):
    plt.show()
if(save_AUC_chart):
    plt.savefig(title+".png")
print("Machine accuracy: " + str(acc))
print("Machine FPR: " + str(fp/len(labels_normal)))
print("Machine TPR: " + str(tp/len(labels_cancer)))
print("Machine AUC: " + str(roc_auc))

ROCs = []
for iteration in range(1000):
    indices = resample(range(len(predictions)), random_state=iteration)
    sample_predictions = predictions[indices]
    sample_scores = conf_roc[indices]
    sample_labels = labels_all[indices]
    fpr, tpr, thresholds = roc_curve(sample_labels, sample_scores)
    roc_auc_sample = auc(fpr, tpr)
    ROCs.append(roc_auc_sample)
    if(print_statements_debug):
        print(str(iteration) + " sample AUC: " + str(ROCs[len(ROCs)-1]))

tn, fp, fn, tp = confusion_matrix(labels_all, predictions).ravel()
acc = accuracy_score(labels_all, predictions)

print("Avg AUC: " + str(np.average(ROCs)))
print("STDev of CV AUC: " + str(np.std(ROCs)))

sorted_scores = np.array(ROCs)
sorted_scores.sort()
confidence_lower = sorted_scores[int(0.025 * len(ROCs))]
confidence_upper = sorted_scores[int(0.975 * len(ROCs))]
print("95% CI for CV AUC: " + str(confidence_lower) + " to " + str(confidence_upper))

i = 0
radio_prediction_list = []
radio_conf_list = []
while i < len(names_all):
    radio_prediction_list.append(radio_input_classify[names_all[i]])
    radio_conf_list.append(radio_input_confidence[names_all[i]])
    if(radio_prediction_list[i] == 0):
        radio_conf_list[i] = -radio_conf_list[i]
    i = i + 1

#fpr, tpr, thresholds = roc_curve(labels_all, radio_prediction_list)
#roc_auc = auc(fpr, tpr)
#tn, fp, fn, tp = confusion_matrix(labels_all, radio_prediction_list).ravel()
#acc = accuracy_score(labels_all, radio_prediction_list)
#print("Radiologist accuracy: " + str(acc))
#print("Radiologist FPR: " + str(fp/len(labels_normal)))
#print("Radiologist TPR: " + str(tp/len(labels_cancer)))
#print("Radiologist AUC: " + str(roc_auc))
#plt.hist(ROCs)
#if(show_charts):
#    plt.show()

if(do_TSNE):
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
    tsne_embedding = TSNE(n_components=2, perplexity=60, init='random', learning_rate=200, n_iter=10000, random_state=0).fit_transform(final_values)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(tsne_embedding[0:len(codes_normal),0], tsne_embedding[0:len(codes_normal),1], edgecolors='none', c="blue", label="normal")
    ax.scatter(tsne_embedding[len(codes_normal):,0], tsne_embedding[len(codes_normal):,1], edgecolors='none', c="red", label="cancer")
    plt.legend(loc='lower right', fontsize='x-large')
    plt.title("t-sne embedding")
    plt.xlim([min(tsne_embedding[:,0]-1) - 0.1 *(max(tsne_embedding[:,0]) - min(tsne_embedding[:,0])), max(tsne_embedding[:,0]) + 0.1 *(max(tsne_embedding[:,0]) - min(tsne_embedding[:,0]))])
    plt.ylim([min(tsne_embedding[:,1]-1) - 0.1 *(max(tsne_embedding[:,1]) - min(tsne_embedding[:,1])), max(tsne_embedding[:,1]) + 0.1 *(max(tsne_embedding[:,1]) - min(tsne_embedding[:,1]))])
    if(show_charts):
        plt.show()


if(save_results_and_do_post_hoc_tests):
    with open('../Results/LGBMInceptionV4NoCropSameDirWithRadioGist.txt', 'w') as f:
        for item in ROCs:
            f.write("%s\n" % item)
    """
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

    set_1 = []
    set_2 = []
    set_3 = []

    with open('../Results/LGBMInceptionV4NoCropSameDirNoRadioGist.txt', 'r') as f:
        for x in f:
            set_1.append(float(x))
    with open('../Results/LGBMInceptionV4NoCropSameDirWithRadioGist.txt', 'r') as f:
        for x in f:
            set_2.append(float(x))
    with open('../Results/RadiologistBootstrappingROCs.txt', 'r') as f:
        for x in f:
            set_3.append(float(x))
            
    w, p1 = stats.levene(set_1, set_2, set_3)
    h_value, pvalue = stats.f_oneway(set_1, set_2, set_3)
    print("W: " + str(w))
    print("p: " + str(p1))
    print("F: " + str(h_value))
    print("p: " + str(pvalue))

print("Accuracy\tTPR\tFPR\tAUC\tAUC stdev\t95% CI")
print("{0:.3f}\t\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t\t[{5:.3f},{6:.3f}]".format(round(acc,3), round(tp/len(labels_cancer), 3), round(fp/len(labels_normal), 3), round(roc_auc, 3), round(np.std(ROCs),3), round(confidence_lower,3), round(confidence_upper,3)))
