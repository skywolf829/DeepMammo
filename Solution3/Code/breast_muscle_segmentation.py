import numpy as np
import sys
sys.path.insert(0, './breast_segment/breast_segment')
from breast_segment import breast_segment
from matplotlib import pyplot as plt
import PIL
import cv2
import warnings
from medpy.filter.smoothing import anisotropic_diffusion

def find_start_and_end_points(im):
    start = None
    end = None
    i = im.shape[1] - 1
    while i > 0 and start is None:
        if im[im.shape[0]-1, i] != 0:
            start = (im.shape[0]-1, i)
        i = i - 1
    if start is None:
        i = im.shape[0] - 1
        while i > 0 and start is None:
            if(im[i, 0] != 0):
                start = (i, 0)
            i = i - 1
    i = im.shape[1] - 1
    while i > 0 and end is None:
        if im[0, i] > 0:
            end = (0, i)
        i = i - 1
    return start, end

def canny(im):
    img = np.zeros([im.shape[0],im.shape[1],3])

    img[:,:,0] = im
    img[:,:,1] = im
    img[:,:,2] = im
    img = img.astype(np.uint8)
    edges = cv2.Canny(img, 0, 32)
    """
    plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    """    
    return edges
def count_top_left_neighbors(im, pos):
    toCheck = [pos]
    allSpots = []
    while len(toCheck) > 0:
        p = toCheck.pop(0)
        allSpots.append(p)
        # Check above
        if p[0] > 0 and im[p[0]-1, p[1]] != 0 and (p[0]-1, p[1]) not in allSpots:
            toCheck.append((p[0]-1, p[1]))
        # Check to left
        if p[1] > 0 and im[p[0], p[1]-1] != 0 and (p[0], p[1]-1) not in allSpots:
            toCheck.append((p[0], p[1]-1))
    return allSpots
def count_bot_right_neighbors(im, pos):
    toCheck = [pos]
    allSpots = []
    while len(toCheck) > 0:
        p = toCheck.pop(0)
        allSpots.append(p)
        # Check below
        if p[0] < im.shape[0]-1 and im[p[0]+1, p[1]] != 0 and (p[0]+1, p[1]) not in allSpots:
            toCheck.append((p[0]+1, p[1]))
        # Check to right
        if p[1] < im.shape[1]-1 and im[p[0], p[1]+1] != 0 and (p[0], p[1]+1) not in allSpots:
            toCheck.append((p[0], p[1]+1))
    return allSpots

def clean_canny(im, start, end):
    im_og = np.copy(im)
    m = (start[0] - end[0]) / (end[1] - start[1])
    y_int = m * end[1]
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if im[y, x] != 0 and y > x * -m + y_int:
                im[y, x] = 0
    for y in range(im.shape[0]):
        for x in range(1, im.shape[1]-1):
            if im[y, x] != 0 and im[y, x-1] != 0 and im[y, x+1] != 0:
                im[y, x] = 0
                im[y, x-1] = 0
                im[y, x+1] = 0
    for y in range(1, im.shape[0]-1):
        for x in range(1, im.shape[1]-1):
            if im[y, x] != 0 and (im[y-1, x-1] != 0 and im[y+1, x+1] != 0 and im[y-1, x+1] != 0) or (im[y-1, x-1] != 0 and im[y+1, x+1] != 0 and im[y+1, x-1] != 0) or (im[y-1, x-1] != 0 and im[y+1, x-1] != 0 and im[y-1, x+1] != 0) or (im[y-1, x+1] != 0 and im[y+1, x+1] != 0 and im[y+1, x-1] != 0):
                top_left = count_top_left_neighbors(im, (y, x))
                bot_right = count_bot_right_neighbors(im, (y, x))
                for spot in top_left:
                    im[spot[0], spot[1]] = 0
                for spot in bot_right:
                    im[spot[0], spot[1]] = 0
    plt.subplot(121)
    plt.imshow(im_og,cmap = 'gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(im,cmap = 'gray')
    plt.title('Edge Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return im
def load_im(path):
    im = PIL.Image.open(path).convert('L')
    side = "R"
    if path.split("_")[1].split(".")[0] == "L":
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        side = "L"
    return np.array(im), side


def segment_pectoral_from_breast(path):
    im, side = load_im(path)
    mask, bbox = breast_segment(im, scale_factor=1)
    #f, ax = plt.subplots(1, figsize=(8, 8))
    #ax.imshow(mask, vmin=0, vmax=1, cmap='gray')
    #plt.show()
    im = np.multiply(im, mask)
    start, end = find_start_and_end_points(mask)
    print(start)
    print(end)
    im = np.array(PIL.Image.fromarray(im).filter(PIL.ImageFilter.MedianFilter(size=33)))
    im = anisotropic_diffusion(im, niter=1)
    edges = canny(im)
    edges = clean_canny(edges, start, end)
segment_pectoral_from_breast("../Images/NORMAL/N3_L.bmp")

