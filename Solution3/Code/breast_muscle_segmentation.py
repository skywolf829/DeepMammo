import numpy as np
import sys
sys.path.insert(0, './breast_segment/breast_segment')
from breast_segment import breast_segment
from matplotlib import pyplot as plt
import PIL
import cv2
import warnings
from medpy.filter.smoothing import anisotropic_diffusion
import math
import random


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
    img = np.zeros([im.shape[0],im.shape[1],1])

    img[:,:,0] = im
    #img[:,:,1] = im
    #img[:,:,2] = im
    img = img.astype(np.uint8)
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    edges = cv2.Canny(thresh_im, lowThresh, high_thresh)
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
def return_connected_components(im):
    components = []
    checked = []
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if im[y, x] != 0 and (y, x) not in checked:
                component, wasChecked = get_component(im, (y, x))
                for item in wasChecked:
                    checked.append(item)
                components.append(component)
    return components

def get_component(im, spot):
    component = []
    checked = []
    toCheck = [spot]
    while len(toCheck) != 0:
        s = toCheck.pop(0)
        checked.append(s)
        component.append(s)
        neighbors = get_nonzero_neighbors(im, s)
        for n in neighbors:
            if n not in checked and n not in toCheck:
                toCheck.append(n)
    return component, checked
def find_greatest_dist(points):
    greatestDist = ((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) ** 0.5
    ends = [points[0], points[1]]
    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            d = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
            if greatestDist < d:
                greatestDist = d
                ends = [points[i], points[j]]
    return ends
def get_slope(im, component):
    ends = []
    for item in component:
        if len(get_nonzero_neighbors(im, item)) == 1:
            ends.append(item)
    if len(ends) > 2:
        ends = find_greatest_dist(ends)
    slope = None
    theEnd = None
    if len(ends) < 2:
        slope = None
    elif ends[0][1] - ends[1][1] == 0:
        slope = 0
    elif ends[0][0] - ends[1][0] == 0:
        slope = None
    else:    
        slope = (ends[0][0] - ends[1][0]) / (ends[0][1] - ends[1][1])
        theEnd = ends[0]
    return slope, theEnd
def color_image_components(im, components):
    im_color = np.zeros([im.shape[0],im.shape[1],3])

    im_color[:,:,0] = im
    im_color[:,:,1] = im
    im_color[:,:,2] = im
    
    for component in components:
        c = [random.random(), random.random(), random.random()]
        for item in component:
            im_color[item] = c
    cv2.imshow('image',im_color)
    cv2.waitKey(0)


def get_nonzero_neighbors(im, spot):
    neighbors = []
    if spot[0] > 0 and im[spot[0]-1, spot[1]] != 0:
        neighbors.append((spot[0]-1, spot[1]))
    if spot[1] > 0 and im[spot[0], spot[1]-1] != 0:
        neighbors.append((spot[0], spot[1]-1))
    if spot[0] < im.shape[0]-1 and im[spot[0]+1, spot[1]] != 0:
        neighbors.append((spot[0]+1, spot[1]-1))
    if spot[1] < im.shape[1]-1 and im[spot[0], spot[1]+1] != 0:
        neighbors.append((spot[0], spot[1]+1))

    if spot[0] > 0 and spot[1] > 0 and im[spot[0]-1, spot[1]-1] != 0:
        neighbors.append((spot[0]-1, spot[1]-1))
    if spot[0] > 0 and spot[1] < im.shape[1]-1 and im[spot[0]-1, spot[1]+1] != 0:
        neighbors.append((spot[0]-1, spot[1]+1))
    if spot[0] < im.shape[0]-1 and spot[1] > 0 and im[spot[0]+1, spot[1]-1] != 0:
        neighbors.append((spot[0]+1, spot[1]-1))
    if spot[0] < im.shape[0]-1 and spot[1] < im.shape[1]-1 and im[spot[0]+1, spot[1]+1] != 0:
        neighbors.append((spot[0]+1, spot[1]+1))
    
    return neighbors
def return_bad_components(im, components, x_int, y_int):
    bad_components = []
    for component in components:
        if len(component) < 2:
            bad_components.append(component)
        else:
            m, point = get_slope(im, component)
            if m is None:
                bad_components.append(component)
            elif m == 0:
                bad_components.append(component)
            """elif math.degrees(math.atan(-1 * m)) > 90 or math.degrees(math.atan(-1 * m)) < 10:
                bad_components.append(component)"""
            """else:
                component_x_int = point[1] - (point[0] / m)
                component_y_int = -1 * m * point[1] + point[0]
                if component_x_int > x_int or component_y_int > y_int:
                    bad_components.append(component)"""
    return bad_components

def clean_canny(im, start, end):
    im_og = np.copy(im)
    m = (start[0] - end[0]) / (end[1] - start[1])
    if start[1] > end[1]:
        m = -m
    y_int = m * end[1]
    # Remove lines beyond the breast boundary
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if im[y, x] != 0 and y > x * -m + y_int:
                im[y, x] = 0
    # Remove horizontal lines
    for y in range(im.shape[0]):
        for x in range(1, im.shape[1]-1):
            if im[y, x] != 0 and im[y, x-1] != 0 and im[y, x+1] != 0:
                im[y, x-1] = 0
                im[y, x+1] = 0
    # Remove diagonal lines
    for y in range(1, im.shape[0]-1):
        for x in range(1, im.shape[1]-1):
            if im[y, x] != 0 and (im[y-1, x-1] != 0 and im[y+1, x+1] != 0 and im[y-1, x+1] != 0) or (im[y-1, x-1] != 0 and im[y+1, x+1] != 0 and im[y+1, x-1] != 0) or (im[y-1, x-1] != 0 and im[y+1, x-1] != 0 and im[y-1, x+1] != 0) or (im[y-1, x+1] != 0 and im[y+1, x+1] != 0 and im[y+1, x-1] != 0):
                top_left = count_top_left_neighbors(im, (y, x))
                bot_right = count_bot_right_neighbors(im, (y, x))
                if len(top_left) >= 3:
                    im[y-1, x-1] = 0
                if len(bot_right) >= 3:
                    im[y+1, x+1] = 0
    # Remove half-bullnose + bullnose
    for y in range(1, im.shape[0]-1):
        for x in range(1, im.shape[1]-1):
            if im[y, x] != 0 and im[y+1, x+1] != 0:
                bot_right = count_bot_right_neighbors(im, (y, x))
                if len(bot_right) >= 3:
                    im[y+1, x+1] = 0     
    components = return_connected_components(im)
    x_int = end[1]
    im_before_component_removal = im.copy()
    color_image_components(im, components)
    bad_components = return_bad_components(im, components, x_int, y_int)
    for component in bad_components:
        for item in component:
            im[item[0], item[1]] = 0
    plt.subplot(121)
    plt.imshow(im_og,cmap = 'gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(121)
    plt.imshow(im_before_component_removal,cmap = 'gray')
    plt.title('Edge Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(im,cmap = 'gray')
    plt.title('Final')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return im
def load_im(path):
    im = PIL.Image.open(path).convert('L')
    side = path.split("_")[1].split(".")[0]
    if side == "L":
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return np.array(im), side


def segment_pectoral_from_breast(path):
    im, side = load_im(path)
    mask, bbox = breast_segment(im, scale_factor=1)
    #f, ax = plt.subplots(1, figsize=(8, 8))
    #ax.imshow(mask, vmin=0, vmax=1, cmap='gray')
    #plt.show()
    im = np.multiply(im, mask)
    im = np.array(PIL.Image.fromarray(im).resize((224, 224)))
    start, end = find_start_and_end_points(im)
    print(start)
    print(end)
    im = np.array(PIL.Image.fromarray(im).filter(PIL.ImageFilter.MedianFilter(size=9)))
    im = anisotropic_diffusion(im, niter=1)
    edges = canny(im)
    edges = clean_canny(edges, start, end)
segment_pectoral_from_breast("../Images/NORMAL/N14_R.bmp")

