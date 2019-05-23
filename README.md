# DeepMammo
This is the third solution to using deep learning and transfer learning for detecting breast cancer in mammography. This solution intends to incorporate radiologist input into the network to 1) allow humans to help the machine learn and 2) allow machines to point out areas of interest for humans.

## Getting started
1. Clone repository with
~~~~
git clone https://github.com/skywolf829/DeepMammo
~~~~

2. Navigate to the DeepMammo/Solution3/Code folder.

3. Clone tensorflow-vgg with
~~~~
git clone https://github.com/machrisaa/tensorflow-vgg
~~~~

4. Rename tensorflow-vgg to tensorflowvgg with 
~~~~
mv tensorflow-vgg tensorflowvgg
~~~~

5. Naivgate into Solution3/Code/tensorflowvgg and open vgg19.py. Edit line 20
~~~~
self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
~~~~
to look like 
~~~~
self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
~~~~

6. Navigate to Solution3/Code and clone tensorflow models with
~~~~
git clone https://github.com/tensorflow/models/
~~~~

7. Install python packages with
~~~~
pip install argparse sklearn keras tensorflow numpy matplotlib
~~~~

8. Download the Inception v4 pre-trained model from https://github.com/tensorflow/models/tree/master/research/slim. Place it in the Solution3/Code directory.

9. Download the VGG19 pre-trained model from https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs. Place this in Solution3/Code/tensorflowvgg.

## Files

### classifier.py
Used to create a classifier utilizing VGG19 with a linear SVM for transfer learning. In lines 21 and 22, decide which folders will be for the two classes (see lines 15-17 for options). Run this code with 
~~~~
python classifier.py
~~~~
The saved codes and labels will be dumped in the directory the script is in as codes.npy and labels.npy. This script may take a few minutes to run depending on the machine. My office machine with a Quadro M5000 and two E5-2630 v4 10 core CPU's takes 2:30 to run this script with normal tensorflow, and 1:30 to run this script with tensorflow-gpu. Another machine with a RTX 2080Ti runs this script with tensorflow-gpu in about 45 seconds.
Script will output predictions and accuracy on a 20% test split at the end.

### classifier_auc.py
Used to create a classifier. utilizing VGG19 with a linear SVM for transfer learning. In lines 24 and 25, decide which folders will be for the two calsses (see lines 18-20 for options). Run this code with
~~~~
python classifier_auc.py
~~~~
The saved codes and labels will be dumped in the directory the script is in as codes.npy and labels.npy. This script may take a few minutes to run depending on the machine, but takes roughly 30 seconds on a 2080Ti.
The script will output guesses as well as create a Area-Under-Curve (AUC) graph at the end for evalutation.

### utility_functions.py
<<<<<<< HEAD
Houses various utility functions that slim down the code in classifier.py and classifier_auc.py. Mainly for loading the images and labels into Nx224x224x3 and Nx1 tensors for VGG19. Also used to create the train/test split necessary for building the TFRecords for the tf-slim related models such as inception.

### build_image_data.py
Originaly taken from models/research/inception/inception/data/build_image_data.py.
Takes folders of images and creates TFRecords for them. Used for the inceptionv4 model available.

To replicate, change lines 79, 81, 83, 101 to the respective file locations for train directory, validation directory, output directory, and labels directory respectively. Examples are:

'../ImagesForTFRecord/TrainingSet/ABNORMALvsCONTRALATERAL'
'../ImagesForTFRecord/TestSet/ABNORMALvsCONTRALATERAL'
'../TFRecords/ABNORMALvsCONTRALATERAL'
'../ImagesForTFRecord/LabelsFiles/ABNORMALvsCONTRALATERAL.txt'

This was repeated for each combination, ABNORMALvsCONTRALATERAL, NORMALvsABNORMAL, NORMALvsCONTRALATERAL.


