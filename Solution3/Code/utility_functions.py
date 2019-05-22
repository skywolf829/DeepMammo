import sys
sys.path.insert(0, '/tensorflowvgg')
import os
import csv
import pickle
from os.path import isfile, isdir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import argparse
import tensorflowvgg.vgg19 as vgg19
from PIL import Image

"""
Takes a directory and splits it into a group of training and testing data.

Inputs:
directory - a single directory to split
test_proportion - float between 0.0 and 1.0 to split the directory into.

Returns:
files_train - list of filepaths for a training set of size len(os.listdir(directory)) * (1-test_proportion)
files_test - list of filepaths for a training set of size len(os.listdir(directory)) * (test_proportion)
        Will be os.listdir(directory) - files_train
fileNames_train - list of filenames for files_train such that the ith entry is the name for the ith index in files_train
fileNames_test - list of filenames for files_test such that the ith entry is the name for the ith index in files_test
"""
def trainTestSplitFromFolder(directory, test_proportion):
    filesList = []
    fileNames = []
    for file_x in os.listdir(directory):
        filesList.append(os.path.join(directory, file_x))
        fileNames.append(os.path.splitext(file_x)[0])
    files_train, files_test, fileNames_train, fileNames_test = train_test_split(filesList, fileNames, test_size=test_proportion)
    return files_train, files_test, fileNames_train, fileNames_test

"""
Takes a list of filepaths and filenames for a train and test set, then saves them into a trainDirectory and testDirectory

Inputs:
files_train - list of filepaths for the training set
files_test - list of filepaths for the test set
fileNames_train - list of file names for the training set. ith entry should correspond with the ith index in files_train
fileNames_test - list of file names for the test set. ith entry should correspond with the ith index in files_test
trainDirectory - where to save the training files
testDirectory - where to save the test files
"""
def dirToTrainTestSplitDirs(files_train, files_test, fileNames_train, fileNames_test, trainDirectory, testDirectory):
    for i in range(len(files_train)):
        img = Image.open(files_train[i])
        img.save(os.path.join(trainDirectory, fileNames_train[i] + ".png"), "png")
    for i in range(len(files_test)):
        img = Image.open(files_test[i])
        img.save(os.path.join(testDirectory, fileNames_test[i] + ".png"),  "png")

"""
Clears all files in the specific folders, and then populates those with an 80/20 train/test split for use by TFRecords
Does this for each type of image (normal, abnormal, contralateral)
"""
def createTestTrainSplitForAllImageTypes():
    deleteFilesInDirectory("../ImagesForTFRecord/TrainingSet/Contralateral")
    deleteFilesInDirectory("../ImagesForTFRecord/TestSet/Contralateral")
    deleteFilesInDirectory("../ImagesForTFRecord/TrainingSet/Abnormal")
    deleteFilesInDirectory("../ImagesForTFRecord/TestSet/Abnormal")
    deleteFilesInDirectory("../ImagesForTFRecord/TrainingSet/Normal")
    deleteFilesInDirectory("../ImagesForTFRecord/TestSet/Normal")
    
    files_train, files_test, fileNames_train, fileNames_test = trainTestSplitFromFolder("../Images/NORMAL", 0.2)
    dirToTrainTestSplitDirs(files_train, files_test, fileNames_train, fileNames_test, "../ImagesForTFRecord/TrainingSet/Normal", "../ImagesForTFRecord/TestSet/Normal")
    files_train, files_test, fileNames_train, fileNames_test = trainTestSplitFromFolder("../Images/CONTRALATERAL BREAST TO CANCEROUS", 0.2)
    dirToTrainTestSplitDirs(files_train, files_test, fileNames_train, fileNames_test, "../ImagesForTFRecord/TrainingSet/Contralateral", "../ImagesForTFRecord/TestSet/Contralateral")
    files_train, files_test, fileNames_train, fileNames_test = trainTestSplitFromFolder("../Images/CANCER", 0.2)
    dirToTrainTestSplitDirs(files_train, files_test, fileNames_train, fileNames_test, "../ImagesForTFRecord/TrainingSet/Abnormal", "../ImagesForTFRecord/TestSet/Abnormal")

"""
Deletes all files in a specific directory. Used in createTestTrainSplitForAllImageTypes.
"""
def deleteFilesInDirectory(directory):
    for file in os.listdir(directory):
        os.remove(file)

"""
Used in classifier.py and classifier_auc.py to load images and labels from multiple directories.

Inputs:
dirs - the folders that hold images of one class
classification - the labels for the ith index of dirs

Returns:
batch - a single batch of size [N, 244, 244, 3] of all images within the directories, with N=num images in each directory within dirs
labels - a list of size [N, 1] where each entry is the class for the ith index of batch
"""
def loadImagesFromDir(dirs, classification):    
    labels = []
    batch = []
    names = []
    for dir_i in range(len(dirs)):
        class_x = classification[dir_i]
        dir_x = dirs[dir_i]
        print("Loading class " + str(class_x) + " from " + str(dir_x))
        files = os.listdir(dir_x)
        for i, file in enumerate(files, 1):
            # Add images to the current batch
            #print(i)
            file_name = os.path.join(dir_x, file)
            #print(file_name)
            img = load_img(file_name, target_size=(224, 224))
            img_arr = img_to_array(img)
            #print("Shape of imgarr:")
            #print(img_arr.shape)
            batch.append(img_arr)  # change according to original script
            #print(np.shape(batch))
            labels.append(class_x)
            names.append(file)
    return batch, labels, names

"""
Loads a csv file with average responses for images to be classified as cancerous or normal.

Input:
file_loc - string of the csv file to read
    Row 1 should be the image names
    Row 2 should be the confidence values where 100 is cancerous and 0 is normal
cancer_label - the label for the decision an image contains cancer
normal_label - the label for the decision an image is normal

Returns:
radio_input_classify - a dictionary mapping from image name -> classification (cancerous or normal)
radio_input_confidence - a dictionary mapping from image name -> confidence in the decision
"""
def loadRadiologistData(file_loc, cancer_label, normal_label):
    radio_input_classify = {}
    radio_input_confidence = {}
    with open(file_loc) as file:
        csvFile = csv.reader(file)
        r = 0
        for row in csvFile:                 
            if r != 0:
                image_name = row[0].replace('-', '_') + ".bmp"
                if image_name != None and image_name != "" and image_name != " ":
                    if image_name not in radio_input_classify:
                        radio_input_classify[image_name] = normal_label
                        radio_input_confidence[image_name] = .5
                    if image_name not in radio_input_confidence:
                        radio_input_classify[image_name] = normal_label
                        radio_input_confidence[image_name] = .5   

                    if row[1] is not "" and row[1] is not " " and row[1] is not None:
                        value = float(row[1])
                        if value < 50:
                            radio_input_classify[image_name] = cancer_label
                            radio_input_confidence[image_name] = (50 - value) / 50
                        else:
                            radio_input_classify[image_name] = normal_label
                            radio_input_confidence[image_name] = (value - 50) / 50
            r = r + 1
    return radio_input_classify, radio_input_confidence

"""
Creates a new array of size Nx4 where the entries are 
    model confidence, model classification, radiologist confidence, radiologist classification

Input:
    img_to_confidence_model - dictionary of filenames -> confidence values for the output of a model
    img_to_classification_model - dictionary of filenames -> classification for the output of a model
    img_to_confidence_radiologist - dictionary of filenames -> confidence values for a radiologist decision
    img_to_classification_radiologist - dictionary of filenames -> classification for the radiologist decision

Returns:
    Nx4 array where each entry is the 4 above values in that order
"""
def createFeaturesFromDicts(img_to_confidence_model, img_to_classification_model, img_to_confidence_radiologist, img_to_classification_radiologist, filenames):
    newInputs = []
    for filename in filenames:
        featureVector = []
        featureVector.append(img_to_classification_model[filename])
        featureVector.append(img_to_confidence_model[filename])
        if filename in img_to_classification_radiologist:
            featureVector.append(img_to_classification_radiologist[filename])
        else:
            featureVector.append(0)    
        if filename in img_to_confidence_radiologist:    
            featureVector.append(img_to_confidence_radiologist[filename])
        else:
            featureVector.append(0)    
        newInputs.append(featureVector)
    return newInputs