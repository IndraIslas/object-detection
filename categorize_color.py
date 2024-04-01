from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import json

def load_color_dict(input_path):
    with open(input_path, 'r') as file:
        color_dict = json.load(file)
    return color_dict

def detect_colors(img_path, color_dict, target_colors_path, bg_color_path):
    ranges = [(0, 63), (63, 126), (126, 189), (189, 252), (252, 255)]
    img = cv.imread(img_path)
    colors = img.reshape(-1, img.shape[-1])
    unique_colors, _ = np.unique(colors, axis=0, return_counts=True)
    print(f'There are {len(unique_colors)} unique colors detected in the image')
    categorized_colors = []
    for color in unique_colors:
        color_bgr = []
        # print(f'color: {color}')
        for c in color:
            for r in ranges:
                if r[0] <= c and c <= r[1]:
                    color_bgr.append(r[0])
                    break
        color_bgr = tuple(color_bgr)
        # print(f'color bgr: {color_bgr}')
        key_name = str(color_bgr)
        # print(f'key name: {key_name}')
        closest_bgr_name = color_dict[key_name]["Name"]
        closest_bgr = color_dict[key_name]["BGR"]
        # print(f'closest bgr name: {closest_bgr_name}')
        categorized_colors.append(closest_bgr)
    categorized_colors = [tuple(color) for color in categorized_colors]
    color_counter = Counter(categorized_colors)
    organized_colors = color_counter.most_common()
    print(f'There are {len(organized_colors)} recognized colors detected in the image')
    if organized_colors:
        bg_color, count = organized_colors[0]
        bg_img = np.full((500, 500, 3), bg_color, dtype=np.uint8)
        cv.imshow(f'Background Color: {bg_color} with {count}', bg_img)
        cv.waitKey(0)
        organized_colors = organized_colors[1:]
        for (color_bgr, count) in organized_colors:
            color_name = color_dict[str(color_bgr)]["Name"]
            color_img = np.full((500, 500, 3), color_bgr, dtype=np.uint8)
            cv.imshow(f'{color_name}: {color_bgr} with {count}', color_img)
            cv.waitKey(0)
        with open(target_colors_path, 'w') as file:
            file.write(str(organized_colors))
        with open(bg_color_path, 'w') as file:
            file.write(str(bg_color, count))

def get_bg_color(img_path, color_dict):
    img = cv.imread(img_path)
    org_bg_color = img[0, 0]
    print(f'Original Background Color 2: {org_bg_color}')
    org_bg_img = np.full((250, 250, 3), org_bg_color, dtype=np.uint8)
    ranges = [(0, 63), (63, 126), (126, 189), (189, 252), (252, 255)]
    closest_bgr = []
    for c in org_bg_color:
        for r in ranges:
            if r[0] <= c and c <= r[1]:
                closest_bgr.append(r[0])
                break
    closest_bgr = tuple(closest_bgr)  
    key_bg = str(tuple(closest_bgr))
    closest_bg_name = color_dict[key_bg]['Name']
    closest_bg_bgr = color_dict[key_bg]['BGR']
    closest_bg = np.full((250, 250, 3), closest_bg_bgr, dtype=np.uint8)
    print(f'Closest Background Color 2: {closest_bg_name}')   
    cv.imshow('Original Background Color 2', org_bg_img)
    cv.waitKey(0)
    cv.imshow(f'Closest Background Color 2: {closest_bg_name}', closest_bg)
    cv.waitKey(0)
    return org_bg_color
    
if __name__ == '__main__':
    # bgr = (142,22,1)
    input_path = 'Dictionaries/bgr_by_key_colors.json'
    img_path = 'Pictures/hands.png'
    img = cv.imread(img_path)
    cv.imshow('Original Image', img)
    cv.waitKey(0)
    target_colors_path = 'Colors/target_colors.txt'
    bg_color_path = 'Colors/bg_color.txt'
    color_dict = load_color_dict(input_path)
    detect_colors(img_path, color_dict, target_colors_path, bg_color_path)
    bg_color = get_bg_color(img_path, color_dict)
