from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from collections import Counter
from sklearn.cluster import KMeans

def load_color_dictionary(filename):
    color_dict = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            parts = line.split(':')
            bgr_str = parts[0].strip('()')
            bgr = tuple(map(int, bgr_str.split(',')))
            color_name = parts[1].strip()
            color_dict[bgr] = color_name
    return color_dict

def show_color(color_dict):
    for bgr, color_name in color_dict.items():
        color_img = np.full((250, 250, 3), bgr, dtype='uint8')
        cv.imshow(color_name, color_img)
        cv.waitKey(0)

def categorize_color(bgr, color_dict):
    # Define the range mapping for each color component
    ranges = [(0, 63), (63, 126), (126, 189), (189, 252)]
    # Function to find the closest range
    closest_bgr = []
    for c in bgr:
        for r in ranges:
            if r[0] <= c and c <= r[1]:
                closest_bgr.append(r[0])
                break
    closest_bgr = tuple(closest_bgr)
    # The color_dict.get() method returns the value for the given key (closest_rgb), or "Not defined" if the key is not found in the dictionary
    closest_bgr_name = color_dict.get(closest_bgr, "Not defined")
    return closest_bgr, closest_bgr_name

if __name__ == '__main__':
    bgr = (105, 105, 15)
    color_dict = load_color_dictionary('bgr_color_dictionary.txt')
    closest_bgr, closest_bgr_name = categorize_color(bgr, color_dict)
    bgr_img = np.full((250, 250, 3), bgr, dtype='uint8')
    closest_bgr_img = np.full((250, 250, 3), closest_bgr, dtype='uint8')
    print(f'The closest color for {bgr} is: {closest_bgr_name}')
    cv.imshow('Original BGR', bgr_img)
    cv.waitKey(0)
    cv.imshow(f'{closest_bgr}: {closest_bgr_name}', closest_bgr_img)
    cv.waitKey(0)

