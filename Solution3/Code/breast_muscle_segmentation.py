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
import statistics
import os

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
    img = img.astype(np.uint8)
    #cv2.imshow("first_image",img)
    #cv2.waitKey(0)

    high_thresh, thresh_im = cv2.threshold(img[:int(img.shape[0]), :int(img.shape[1]/2)], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    edges = cv2.Canny(thresh_im, lowThresh, high_thresh)

    #cv2.imshow("threshimage",thresh_im)
    #cv2.waitKey(0)
    img2 = np.zeros([im.shape[0],im.shape[1],1])
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            img2[i, j,0]=edges[i, j]
    #cv2.imshow("edges",img2)
    #cv2.waitKey(0)
    
    return img2[:,:,0]

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
        if p[0] > 0 and p[1] > 0 and im[p[0]-1, p[1]-1] != 0 and (p[0]-1, p[1]-1) not in allSpots:
            toCheck.append((p[0]-1, p[1]-1))
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
        if p[0] < im.shape[0]-1 and p[1] < im.shape[1]-1 and im[p[0]+1, p[1]+1] != 0 and (p[0]+1, p[1]+1) not in allSpots:
            toCheck.append((p[0]+1, p[1]+1))
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
        if len(get_nonzero_neighbors(im, item)) <= 2:
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
    #cv2.imshow("image",im_color)
    #cv2.waitKey(0)


def get_nonzero_neighbors(im, spot):
    neighbors = []
    if spot[0] > 0 and im[spot[0]-1, spot[1]] != 0:
        neighbors.append((spot[0]-1, spot[1]))
    if spot[1] > 0 and im[spot[0], spot[1]-1] != 0:
        neighbors.append((spot[0], spot[1]-1))
    if spot[0] < im.shape[0]-1 and im[spot[0]+1, spot[1]] != 0:
        neighbors.append((spot[0]+1, spot[1]))
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
            elif math.degrees(math.atan(-1 * m)) > 90 or math.degrees(math.atan(-1 * m)) < 10:
                bad_components.append(component)
            """else:
                component_x_int = point[1] - (point[0] / m)
                component_y_int = -1 * m * point[1] + point[0]
                if component_x_int > x_int or component_y_int > y_int:
                    bad_components.append(component)"""
    return bad_components

def clean_canny(im, start, end):
    if start == None or end == None:
        return
    im_og = np.copy(im)
    if end[1] == start[1]:
        start = (start[0], start[1]+1)
    if end[0] == start[1]:
        start = (start[0]+1, start[1])

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
    """
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
                    """
    components = return_connected_components(im)
    x_int = end[1]
    im_before_component_removal = im.copy()
    color_image_components(im, components)
    bad_components = return_bad_components(im, components, x_int, y_int)
    for component in bad_components:
        for item in component:
            im[item[0], item[1]] = 0
    #cv2.imshow("im_og",im_og)
    #cv2.waitKey(0)
    #cv2.imshow("im_before_component_removal",im_before_component_removal)
    #cv2.waitKey(0)
    #cv2.imshow("im",im)
    #cv2.waitKey(0)
    return im
def get_bounding_box(component):
    left = component[0][1]
    right = component[0][1]
    top = component[0][0]
    bot = component[0][0]
    for item in component:
        if item[0] < top:
            top = item[0]
        if item[0] > bot:
            bot = item[0]
        if item[1] < left:
            left = item[1]
        if item[1] > right:
            right = item[1]
    return (left, right, top, bot)

def load_im(path):
    im = PIL.Image.open(path).convert('L')
    side = path.split("_")[1].split(".")[0]
    if side == "L":
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return np.array(im), side
def pick_best_component(edges):
    pectoral_boundary = None
    components = return_connected_components(edges)
    if len(components) == 0:
        return None
    lengths = []
    E_c = []
    E_x = []
    for i in range(len(components)):
        lengths.append(len(components[i]))
        ends = []
        for item in components[i]:
            if len(get_nonzero_neighbors(edges, item)) <= 2:
                ends.append(item)
        if len(ends) > 2:
            ends = find_greatest_dist(ends)
        dist = ((ends[0][0] - ends[1][0]) ** 2 + (ends[0][1] - ends[1][1]) ** 2) ** 0.5
        if dist > lengths[i]:
            dist = lengths[i]
        E_c.append(dist / lengths[i])
        (l,r,t,b) = get_bounding_box(components[i])
        E_x.append(lengths[i] / abs(r-l)*abs(t-b))
    mean = statistics.mean(lengths)
    std = 0
    if(len(lengths) > 1):
        std = statistics.stdev(lengths)
    T_hat = mean + std
    long_bois = []

    for i in range(len(lengths)):
        if lengths[i] > T_hat:
            long_bois.append(i)
    
    if len(long_bois) == 1:
        pectoral_boundary = components[long_bois[0]]
    elif len(long_bois) > 1:
        best_length = long_bois[0]
        best_E_c = long_bois[0]
        best_E_x = long_bois[0]
        for i in long_bois:
            if lengths[i] > lengths[best_length]:
                best_length = i
            if E_c[i] > E_c[best_E_c]:
                best_E_c = i
            if E_x[i] > E_x[best_E_x]:
                best_E_x = i
        if best_length == best_E_c or best_length == best_E_x:
            pectoral_boundary = components[best_length]
        elif best_E_c == best_E_x:
            pectoral_boundary = components[best_E_c]
        else:
            pectoral_boundary = components[best_length]
    else:
        best_length = 0
        best_E_c = 0
        best_E_x = 0
        for i in range(len(components)):
            if lengths[i] > lengths[best_length]:
                best_length = i
            if E_c[i] > E_c[best_E_c]:
                best_E_c = i
            if E_x[i] > E_x[best_E_x]:
                best_E_x = i
        if best_length == best_E_c or best_length == best_E_x:
            pectoral_boundary = components[best_length]
        elif best_E_c == best_E_x:
            pectoral_boundary = components[best_E_c]
        else:
            pectoral_boundary = None
    return pectoral_boundary

def grow_boundary(im, original, component, start, end):
    new_im = im.copy()
    if start == None or end == None:
        return im
    ends = []
    for item in component:
        if len(get_nonzero_neighbors(im, item)) <= 2:
            ends.append(item)
    if len(ends) > 2:
        ends = find_greatest_dist(ends)
    segment_start = None
    segment_end = None
    if ends[0][0] > ends[1][0]:
        segment_start = ends[0]
        segment_end = ends[1]
    else:
        segment_start = ends[1]
        segment_end = ends[0]
    
    new_points = []
    spot_to_grow = segment_end
    # grow up
    
    while spot_to_grow[0] != 0:
        starting_intensity = get_avg_intensity(original, spot_to_grow, 9)
        best_spot = spot_to_grow[1]
        best_diff = abs(starting_intensity - get_avg_intensity(original, (spot_to_grow[0]-1, spot_to_grow[1]), 9))
        for i in range(spot_to_grow[1]-2, spot_to_grow[1]+2):
            if i >= 0 and i < im.shape[1]:
                spot_intensity = get_avg_intensity(original, (spot_to_grow[0]-1, i), 9)
                diff = abs(starting_intensity - spot_intensity)
                if diff < best_diff:
                    best_diff = diff
                    best_spot = i
        new_points.append((spot_to_grow[0]-1, best_spot))
        spot_to_grow = (spot_to_grow[0]-1, best_spot)
    
    spot_to_grow = segment_start
    # grow down
    while spot_to_grow[1] != 0 and spot_to_grow[0] != im.shape[0] - 1:
        starting_intensity = get_avg_intensity(original, spot_to_grow, 9)
        best_spot = spot_to_grow[1]
        best_diff = abs(starting_intensity - get_avg_intensity(original, (spot_to_grow[0]+1, spot_to_grow[1]), 9))
        for i in range(spot_to_grow[1]-2, spot_to_grow[1]+2):
            if i >= 0 and i < im.shape[1]:
                spot_intensity = get_avg_intensity(original, (spot_to_grow[0]+1, i), 9)
                diff = abs(starting_intensity - spot_intensity)
                if diff < best_diff:
                    best_diff = diff
                    best_spot = i
        new_points.append((spot_to_grow[0]+1, best_spot))
        spot_to_grow = (spot_to_grow[0]+1, best_spot)
    
    for item in new_points:
        new_im[item[0], item[1]] = 1
    return new_im

def get_avg_intensity(im, spot, boxsize):
    intensity = 0
    count = 0
    for i in range(int(spot[0] - boxsize/2), int(spot[0] + boxsize/2)):
        for j in range(int(spot[1]-boxsize/2), int(spot[1]+boxsize/2)):
            if i >= 0 and i < im.shape[0]:
                if j >= 0 and j < im.shape[1]:
                    intensity = intensity + im[i,j]
                    count = count + 1
    if count != 0:
        intensity = intensity / count
    return intensity

def fill_image(im):
    img = im.copy()
    for i in range(img.shape[0]):
        j = 0
        while j < img.shape[1] and img[i,j] == 0:
            j = j + 1
        if j < img.shape[1]:
            for k in range(j):
                img[i,k] = 1
    return img

def finalize_boundary(edges, original, start, end):
    if start == None or end == None:
        return np.ones_like(original)
    
    component = pick_best_component(edges)
    if component is None:
        return np.ones_like(original)
    im = np.zeros_like(edges)
    for item in component:
        im[item[0], item[1]] = 1
    #cv2.imshow("finalcomponent",im)
    #cv2.waitKey(0)
    im = grow_boundary(im, original, component, start, end)
    #cv2.imshow("finalgrowth",im)
    #cv2.waitKey(0)
    im_filled = fill_image(im)
    #cv2.imshow("final_filled",im_filled)
    #cv2.waitKey(0)
    for i in range(im_filled.shape[0]):
        for j in range(im_filled.shape[1]):
            im_filled[i,j] = 1 - im_filled[i,j]
    return im_filled

def segment_pectoral_from_breast(path):
    im_og, side = load_im(path)
    mask, bbox = breast_segment(im_og, scale_factor=1)

    im_og = np.multiply(im_og, mask)
    im = np.array(PIL.Image.fromarray(im_og).resize((int(im_og.shape[1]*0.25), int(im_og.shape[0]*0.25))))
    start, end = find_start_and_end_points(im)
    
    print("start: " + str(start))
    print("end: " + str(end))
    im = np.array(PIL.Image.fromarray(im).filter(PIL.ImageFilter.MedianFilter(size=9)))
    im = anisotropic_diffusion(im, niter=1)
    im = im.astype(np.uint8)
    edges = canny(im)
    #edges = np.array(PIL.Image.fromarray(edges).resize((int(edges.shape[1]*0.25), int(edges.shape[0]*0.25)), PIL.Image.ANTIALIAS))
    edges = clean_canny(edges, start, end)
    final_mask = finalize_boundary(edges, im, start, end)
    
    #cv2.imshow("final_mask",final_mask)
    #cv2.waitKey(0)
    final_image = np.multiply(im_og.astype(np.uint8), np.array(PIL.Image.fromarray(final_mask).resize((im_og.shape[1], im_og.shape[0]))).astype(np.uint8))
    #cv2.imshow("final_image",final_image)
    #cv2.waitKey(0)
    return final_image

def save_all_crops(dir, saveDir):
    for im_name in os.listdir(dir):
        im = segment_pectoral_from_breast(os.path.join(dir,im_name))
        short_name = im_name.split(".")[0]
        PIL.Image.fromarray(im).save(os.path.join(saveDir, short_name + ".png"))


save_all_crops("../Images/CONTRALATERAL BREAST TO CANCEROUS/", "../Images/NewCroppingMethod/Contralateral/")
#save_all_crops("../Images/NORMAL/", "../Images/NewCroppingMethod/Normal/")
#save_all_crops("../Images/CANCER/", "../Images/NewCroppingMethod/Cancer/")

