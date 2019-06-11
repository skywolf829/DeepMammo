import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import utility_functions

images_normal_train, labels_normal_train, names_normal_train = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTrain",), (0,))
images_normal_test, labels_normal_test, names_normal_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInputCroppedv5/NormalTest",), (0,))
images_abnormal_train, labels_abnormal_train, names_abnormal_train = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTrain",), (1,))
images_abnormal_test, labels_abnormal_test, names_abnormal_test = utility_functions.loadImagesFromDir(("../Images/CherryPickedWithRadiologistInputCroppedv5/AbnormalTest",), (1,))


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(16):
    theta = theta / 16. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            print("Kernel ")
            print(kernel)
            kernels.append(kernel)
print(np.array(kernels).shape)
shrink = (slice(0, None, 3), slice(0, None, 3))

# prepare reference features
ref_feats = np.zeros((len(names_normal_train), len(kernels), 2), dtype=np.double)
for i in range(len(names_normal_train)):
    feats = compute_feats(images_normal_train[i], kernels)
    print(feats)
    ref_feats[i, :, :] = feats



