import sys
#sys.path.insert(0, '/tensorflowvgg')
import os
import csv
import pickle
from os.path import isfile, isdir
import numpy as np
#import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
#from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import argparse
#import tensorflowvgg.vgg19 as vgg19
from PIL import Image, ImageFilter, ImageChops
import cv2

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


def rotateImages(img_array, degrees, mirrorUpDown, mirrorLeftRight):
    rotated_imgs = []
    for img in img_array:
        img = Image.fromarray(img, 'RGB')
        if(degrees is not None):
            img.rotate(degrees)
        if(mirrorLeftRight):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if(mirrorUpDown):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        rotated_imgs.append(img_to_array(img))
    return rotated_imgs
         
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
                image_name = row[0].replace('-', '_') + ".png"
                if image_name != None and image_name != "" and image_name != " ":
                    #if image_name not in radio_input_classify:
                    #    radio_input_classify[image_name] = normal_label
                    #    radio_input_confidence[image_name] = .5
                    #if image_name not in radio_input_confidence:
                    #    radio_input_classify[image_name] = normal_label
                    #    radio_input_confidence[image_name] = .5   

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

"""
Prints items one by one in the list on new lines. Easy to copy into excel

Inputs:
theList - a list to be printed
"""
def printListInOrder(theList):
    for item in theList:
        print(item)

"""
Prints values from theDictonary in the order of theList, given that theList is a subset of theDictionary.keys

Inputs:
theList - a list of keys for theDictionary
theDictionary - a dictionary to have its values printed
"""
def printDictionaryInOrder(theList, theDictionary):
    for item in theList:
        if item in theDictionary.keys():
            print(theDictionary[item])
        else:
            print("")

"""
Creates bilateral images from unilateral images using PIL

Inputs:
list_1 - list of file locations for one side mammogram
list_2 - list of file locations for other side mammogram
names - list of file locations to save output files

Note that these should all be in the same order. That is, list_1[i] is the same subject as list_2[i] with the name being names[i]
"""
def createBilateralFromUnilateral(list_1, list_2, names):
    
    for i in range(len(list_1)):
        print("Opening " + list_1[i] + " " + list_2[i])

        img_1 = Image.open(list_1[i])
        img_2 = Image.open(list_2[i])
        totalWidth = img_1.size[0] + img_2.size[0]
        height = max([img_1.size[1], img_2.size[1]])
        
        new_im = Image.new('RGBA', (totalWidth, height))
        new_im.paste(img_1, (0,0))
        new_im.paste(img_2, (img_1.size[0],0))    
        new_im.save(names[i] + ".png", 'png')

"""
Used to create bilateral images from folders of unilateral images.
"""
def createNewDirectoryAndMakeBilaterals():
    bilateralImagesPath = "../Images/Bilateral"
    normalBilateralImagePath = "../Images/Bilateral/Normal"
    cancerBilateralImagePath = "../Images/Bilateral/Cancer"
    normalImagePath = "../Images/NORMAL"
    cancerImagePath = "../Images/CANCER"
    contralateralImagePath = "../Images/CONTRALATERAL BREAST TO CANCEROUS"

    if not os.path.exists(bilateralImagesPath):
        os.mkdir(bilateralImagesPath)
        print("Directory " , bilateralImagesPath ,  " Created ")
    else:    
        print("Directory " , bilateralImagesPath ,  " already exists")

    if not os.path.exists(normalBilateralImagePath):
        os.mkdir(normalBilateralImagePath)
        print("Directory " , normalBilateralImagePath ,  " Created ")
    else:    
        print("Directory " , normalBilateralImagePath ,  " already exists")

    if not os.path.exists(cancerBilateralImagePath):
        os.mkdir(cancerBilateralImagePath)
        print("Directory " , cancerBilateralImagePath ,  " Created ")
    else:    
        print("Directory " , cancerBilateralImagePath ,  " already exists")

    normalImagePaths = {}
    cancerImagePaths = {}
    for item in os.listdir(normalImagePath):
        number = int(item.split("_")[0].split("N")[1])
        letter = item.split("_")[1].split(".")[0]
        if number not in normalImagePaths.keys():
            normalImagePaths[number] = {}
        normalImagePaths[number][letter] = os.path.join(normalImagePath, item)
    for item in os.listdir(cancerImagePath):
        number = int(item.split("_")[0].split("D")[1])
        letter = str(item.split("_")[1].split(".")[0])
        if number not in cancerImagePaths.keys():
            cancerImagePaths[number] = {}
        cancerImagePaths[number][letter] = os.path.join(cancerImagePath, item)
    for item in os.listdir(contralateralImagePath):
        number = int(item.split("_")[0].split("D")[1])
        letter = str(item.split("_")[1].split(".")[0])
        if number not in cancerImagePaths.keys():
            cancerImagePaths[number] = {}
        cancerImagePaths[number][letter] = os.path.join(contralateralImagePath, item)
    normalNames = []
    cancerNames = []
    leftSideNormal = []
    rightSideNormal = []
    leftSideCancer = []
    rightSideCancer = []

    for item in normalImagePaths.keys():
        leftSideNormal.append(normalImagePaths[item]["L"])
        rightSideNormal.append(normalImagePaths[item]["R"])
        normalNames.append(os.path.join(normalBilateralImagePath, "N" + str(item) + "_bilateral"))
    for item in cancerImagePaths.keys():
        leftSideCancer.append(cancerImagePaths[item]["L"])
        rightSideCancer.append(cancerImagePaths[item]["R"])
        cancerNames.append(os.path.join(cancerBilateralImagePath, "AD" + str(item) + "_bilateral"))
    createBilateralFromUnilateral(leftSideNormal, rightSideNormal, normalNames)
    createBilateralFromUnilateral(leftSideCancer, rightSideCancer, cancerNames)

def createRotatedAndMirroredImages(startingImagesDir, newDir):
    for image_name in os.listdir(startingImagesDir):
        print(image_name)
        if len(image_name) < 2:
            continue
        imagePath = os.path.join(startingImagesDir, image_name)
        n = image_name.split('_')[0]
        side = image_name.split('_')[1].split('.')[0]
        image = Image.open(imagePath)
        image_90 = image.rotate(90)
        image_180 = image.rotate(180)
        image_270 = image.rotate(270)
        image_flip_LR = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_flip_UD = image.transpose(Image.FLIP_TOP_BOTTOM)
        image_90_flip_LR = image_90.transpose(Image.FLIP_LEFT_RIGHT)
        image_90_flip_UD = image_90.transpose(Image.FLIP_TOP_BOTTOM)
        image_180_flip_LR = image_180.transpose(Image.FLIP_LEFT_RIGHT)
        image_180_flip_UD = image_180.transpose(Image.FLIP_TOP_BOTTOM)
        image_270_flip_LR = image_270.transpose(Image.FLIP_LEFT_RIGHT)
        image_270_flip_UD = image_270.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(os.path.join(newDir, n + "000"+"_"+side+".png"))
        image_90.save(os.path.join(newDir, n+"001"+"_"+side+".png"))
        image_180.save(os.path.join(newDir, n+"002"+"_"+side+".png"))
        image_270.save(os.path.join(newDir, n+"003"+"_"+side+".png"))
        image_flip_LR.save(os.path.join(newDir, n+"004_"+side+".png"))
        image_flip_UD.save(os.path.join(newDir, n+"005_"+side+".png"))
        image_90_flip_LR.save(os.path.join(newDir, n+"006_"+side+".png"))
        image_90_flip_UD.save(os.path.join(newDir, n+"007_"+side+".png"))
        image_180_flip_LR.save(os.path.join(newDir, n+"008_"+side+".png"))
        image_180_flip_UD.save(os.path.join(newDir, n+"009_"+side+".png"))
        image_270_flip_LR.save(os.path.join(newDir, n+"010_"+side+".png"))
        image_270_flip_UD.save(os.path.join(newDir, n+"011_"+side+".png"))

def removeMassesNotOnSides(image):
    image = np.array(image)
    status = {}
    toCheck = []
    newImage = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            status[(i, j)] = False

    for i in range(image.shape[0]):
        toCheck.append((i, 0))
        toCheck.append((i, image.shape[1]-1))
    for i in range(image.shape[1] - 2):
        toCheck.append((0, i+1))
        toCheck.append((image.shape[0]-1, i+1))

    while(len(toCheck) > 0):
        spot = toCheck.pop(0)
        #print("Checking " + str(spot))
        if not status[spot]:
            #print("Spot not checked yet.")
            status[spot] = True
            if image[spot] > 20:
                newImage[spot[0], spot[1]] = image[spot[0], spot[1]]
                if(spot[0] > 0):
                    toCheck.append((spot[0] - 1, spot[1]))
                if(spot[0] < image.shape[0]- 1):
                    toCheck.append((spot[0] + 1, spot[1]))
                if(spot[1] > 0):
                    toCheck.append((spot[0], spot[1] - 1))
                if(spot[1] < image.shape[1] - 1):
                    toCheck.append((spot[0], spot[1] + 1))
        #else:
            #print("Spot checked")
    return Image.fromarray(np.uint8(newImage))

def cropImageTest(image_path, final_image_path):
    image = Image.open(image_path).convert('L')
    #image.save('test_start.png')
    image = image.filter(ImageFilter.BoxBlur(9)) #3
    image = image.filter(ImageFilter.MaxFilter(size=3)) #3
    image = image.filter(ImageFilter.MinFilter(size=17)) #17
    image = image.filter(ImageFilter.MinFilter(size=9)) #9
    
    image = image.filter(ImageFilter.MaxFilter(size=3)) #3
    image = image.filter(ImageFilter.MaxFilter(size=9)) #3
    image = np.array(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):                        
            #image[x, y] = min((image[x, y], 150))
            if image[x, y] < 100:
                image[x, y] = 0            
    image = Image.fromarray(np.uint8(image))
    #image.save('test_000mind.png')
    i = 1
    #image = image.filter(ImageFilter.MinFilter(size=9))
    for i in range(2):                       
        image = image.filter(ImageFilter.MedianFilter(size=17))
        #if i % 5 == 0:
        #    if i % 10 == 0:
        #        image = image.filter(ImageFilter.MinFilter(size=9))
        #    else:
        #        image = image.filter(ImageFilter.MaxFilter(size=9))
        #image.save('test_0_' + str(i)+'.png')   
    image = np.array(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):                        
            #image[x, y] = min((image[x, y], 150))
            if image[x, y] < 100:
                image[x, y] = 0  
            else:
                image[x, y] = 255
    image = Image.fromarray(np.uint8(image))
    image = removeMassesNotOnSides(image)
    #image.save('test_final.png')
    image = ImageChops.invert(image)
    image = np.array(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):                
            if image[x, y] < 150:
                image[x, y] = 0
            else:
                image[x, y] = 255
    image = Image.fromarray(np.uint8(image))
    image = image.filter(ImageFilter.MinFilter(size=33))
    #image.save('test_final_invert.png')
    image_og = Image.open(image_path).convert('L')
    image = ImageChops.multiply(image, image_og)
    image.save(final_image_path)

def cropAllImagesInDirToDir(original_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Directory " , save_dir ,  " Created ")
    else:    
        print("Directory " , save_dir ,  " already exists")

    for image_name in os.listdir(original_dir):
        print("Cropping " + str(image_name))
        imageShortName = image_name.split(".")[0]
        new_path = os.path.join(save_dir, imageShortName + ".png")
        cropImageTest(os.path.join(original_dir, image_name), new_path)
        print("Finished cropping " + str(image_name))

def split_balanced(data, target, test_size=0.2):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train,:]
    X_test = data[ix_test,:]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test
"""
After realizing that we lacked radiologist input for a large number of the images (88 of the 220 total), we had to change the train/test
split so that the test set all had radiologist input for analysis purposes. 

Run this code once to populate 6 directories inside ../Images/CherryPickedWithRadiologistInput for a train/test split 
for normal, abnormal, and contralateral images
"""
def trainTestSplitWithRadioInput():
    radio_input_classify, radio_input_confidence = loadRadiologistData("../RadiologistData/radiologistInput.csv", 1, 0)
    split_proportion = 0.3
    normal_count = 0
    abnormal_count = 0
    contralateral_count = 0
    normal_selected = 0
    abnormal_selected = 0
    contralateral_selected = 0

    normal_with_radiologistInput = []
    normal_without_radiologistInput = []
    abnormal_with_radiologistInput = []
    abnormal_without_radiologistInput = []
    contralateral_with_radiologistInput = []
    contralateral_without_radiologistInput = []

    
    for image_name in os.listdir("../Images/RandomRectCropForAnalysis/Normal"):
        if image_name in radio_input_classify.keys():
            normal_with_radiologistInput.append(os.path.join("../Images/RandomRectCropForAnalysis/Normal", image_name))
        else:
            normal_without_radiologistInput.append(os.path.join("../Images/RandomRectCropForAnalysis/Normal", image_name))
        normal_count = normal_count + 1

    for image_name in os.listdir("../Images/RandomRectCropForAnalysis/Cancer"):
        if image_name in radio_input_classify.keys():
            abnormal_with_radiologistInput.append(os.path.join("../Images/RandomRectCropForAnalysis/Cancer", image_name))
        else:
            abnormal_without_radiologistInput.append(os.path.join("../Images/RandomRectCropForAnalysis/Cancer", image_name))
        abnormal_count = abnormal_count + 1

    for image_name in os.listdir("../Images/RandomRectCropForAnalysis/Contralateral"):
        if image_name in radio_input_classify.keys():
            contralateral_with_radiologistInput.append(os.path.join("../Images/RandomRectCropForAnalysis/Contralateral", image_name))
        else:
            contralateral_without_radiologistInput.append(os.path.join("../Images/RandomRectCropForAnalysis/Contralateral", image_name))
        contralateral_count = contralateral_count + 1

    normal_needed = (normal_count + abnormal_count) * split_proportion * 0.5
    abnormal_needed = (abnormal_count + normal_count) * split_proportion * 0.5
    contralateral_needed = contralateral_count * split_proportion

    from sklearn.utils import shuffle
    normal_with_radiologistInput = shuffle(normal_with_radiologistInput, random_state=9861350)
    abnormal_with_radiologistInput = shuffle(abnormal_with_radiologistInput, random_state=2896614)
    contralateral_with_radiologistInput = shuffle(contralateral_with_radiologistInput, random_state=4570913)

    while normal_selected < normal_needed and len(normal_with_radiologistInput) > 0:
        im_location = normal_with_radiologistInput.pop(0)
        im_name = im_location.split("\\")[1].split(".")[0]
        im = Image.open(im_location)        
        im.save(os.path.join("../Images/CherryPickedWithRadiologistInputRandomRectCropped/NormalTest/", im_name + ".png"))
        normal_selected = normal_selected + 1
    while len(normal_with_radiologistInput) > 0:
        normal_without_radiologistInput.append(normal_with_radiologistInput.pop(0))
    while len(normal_without_radiologistInput) > 0:
        im_location = normal_without_radiologistInput.pop(0)
        im_name = im_location.split("\\")[1].split(".")[0]
        im = Image.open(im_location)        
        im.save(os.path.join("../Images/CherryPickedWithRadiologistInputRandomRectCropped/NormalTrain/", im_name + ".png"))
    print(str(normal_selected) + " normal images with radiologist input saved to ../Images/CherryPickedWithRadiologistInputRandomRectCropped/NormalTest")
    print(str(normal_count - normal_selected) + " normal images with/without radiologist input saved to ../Images/CherryPickedWithRadiologistInputRandomRectCropped/NormalTrain")

    while abnormal_selected < abnormal_needed and len(abnormal_with_radiologistInput) > 0:
        im_location = abnormal_with_radiologistInput.pop(0)
        im_name = im_location.split("\\")[1].split(".")[0]
        im = Image.open(im_location)        
        im.save(os.path.join("../Images/CherryPickedWithRadiologistInputRandomRectCropped/AbnormalTest/", im_name + ".png"))
        abnormal_selected = abnormal_selected + 1
    while len(abnormal_with_radiologistInput) > 0:
        abnormal_without_radiologistInput.append(abnormal_with_radiologistInput.pop(0))
    while len(abnormal_without_radiologistInput) > 0:
        im_location = abnormal_without_radiologistInput.pop(0)
        im_name = im_location.split("\\")[1].split(".")[0]
        im = Image.open(im_location)        
        im.save(os.path.join("../Images/CherryPickedWithRadiologistInputRandomRectCropped/AbnormalTrain/", im_name + ".png"))
    print(str(abnormal_selected) + " abnormal images with radiologist input saved to ../Images/CherryPickedWithRadiologistInputRandomRectCropped/AbnormalTest")
    print(str(abnormal_count - abnormal_selected) + " abnormal images with/without radiologist input saved to ../Images/CherryPickedWithRadiologistInputRandomRectCropped/AbnormalTrain")

    while contralateral_selected < contralateral_needed and len(contralateral_with_radiologistInput) > 0:
        im_location = contralateral_with_radiologistInput.pop(0)
        im_name = im_location.split("\\")[1].split(".")[0]
        im = Image.open(im_location)        
        im.save(os.path.join("../Images/CherryPickedWithRadiologistInputRandomRectCropped/ContralateralTest/", im_name + ".png"))
        contralateral_selected = contralateral_selected + 1
    while len(contralateral_with_radiologistInput) > 0:
        contralateral_without_radiologistInput.append(contralateral_with_radiologistInput.pop(0))
    while len(contralateral_without_radiologistInput) > 0:
        im_location = contralateral_without_radiologistInput.pop(0)
        im_name = im_location.split("\\")[1].split(".")[0]
        im = Image.open(im_location)        
        im.save(os.path.join("../Images/CherryPickedWithRadiologistInputRandomRectCropped/ContralateralTrain/", im_name + ".png"))
    print(str(contralateral_selected) + " contralateral images with radiologist input saved to ../Images/CherryPickedWithRadiologistInputRandomRectCropped/ContralateralTest")
    print(str(contralateral_count - contralateral_selected) + " contralateral images with/without radiologist input saved to ../Images/CherryPickedWithRadiologistInputRandomRectCropped/ContralateralTrain")

#cropImageTest("../Images/CANCER/AD22_L.bmp", "test_final_masked.png")
#cropAllImagesInDirToDir("../Images/CANCER", "../Images/Cropped/Cancer_newfilters")
#createRotatedAndMirroredImages("../Images/CANCER", "../Images/FlippedAndRotated/Cancer")
#createRotatedAndMirroredImages("../Images/NORMAL", "../Images/FlippedAndRotated/Normal")
#createRotatedAndMirroredImages("../Images/CONTRALATERAL BREAST TO CANCEROUS", "../Images/FlippedAndRotated/Contralateral")
#trainTestSplitWithRadioInput()