from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from collections import Counter
from sklearn.cluster import KMeans
import json

# import numpy as np
# import cv2 as cv

# def load_color_dictionary(file_path):
#     color_dict = {}
#     with open(file_path, 'r') as file:
#         for line in file.readlines():
#             parts = line.strip().split(': ')
#             if len(parts) != 2:
#                 continue
#             # Convert the key from string to a tuple of integers
#             key = tuple(map(int, parts[0][1:-1].split(', ')))
#             color_dict[key] = parts[1]
#     return color_dict

# def find_closest_color(color, color_dict):
#     closest_color = min(color_dict.keys(), key=lambda k: np.linalg.norm(np.array(k) - np.array(color)))
#     return color_dict[closest_color]

# def detect_colors(img_path, color_dict):
#     img = cv.imread(img_path)
#     colors = img.reshape(-1, img.shape[-1])
#     unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
#     categorized_colors = {}
#     for color in unique_colors:
#         category = find_closest_color(color, color_dict)
#         if category not in categorized_colors:
#             categorized_colors[category] = 0
#         categorized_colors[category] += np.sum(counts[np.all(unique_colors == color, axis=1)])
#     for category, count in categorized_colors.items():
#         print(f'Category: {category}, Count: {count}')
#     background_color = unique_colors[np.argmax(counts)]
#     background_category = find_closest_color(background_color, color_dict)
#     print(f'Background Color Category: {background_category}')
#     return categorized_colors, background_category
# color_dict = load_color_dictionary('color_dictionary.txt')

# def show_colors(categorized_colors):
#     for color in categorized_colors:
#         color_img = np.full((100, 100, 3), color, dtype='uint8')
#         cv.imshow(f'{color}', color_img)
#         cv.waitKey(0)

# if __name__ == '__main__':
#     img_path = 'Pictures/color_splashes.png'
#     img = cv.imread(img_path)
#     cv.imshow('Image', img)
#     categorized_colors = detect_colors(img_path, color_dict)
#     show_colors(categorized_colors)
#     print('DONE --------------------')

# # def detect_colors(img_path):
# #     img = cv.imread(img_path)
# #     colors = img.reshape(-1, img.shape[-1])
# #     unique_colors = np.unique(colors, axis=0)
# #     background_color = np.argmax(unique_colors)
# #     print(f'Background Color: {background_color}')
# #     return unique_colors

# # def show_colors(color_list):
# #     for color in color_list:
# #         print(color)
# #         color_img = np.full((100, 100, 3), color, dtype='uint8')
# #         cv.imshow(f'{color}', color_img)
# #         cv.waitKey(0)

# # def multi_color_threshold_mask(img, target_colors, target_threshold):
# #     """
# #     Process an image to make all colors not close to any target colors white, and colors close to any of the target colors black.
# #     Pixels close to ignore_colors are set to white.
# #     :param img: The input image as a NumPy array (BGR format).
# #     :param target_colors: A list of BGR colors to compare against (as [[B, G, R], ...]).
# #     :param target_threshold: A higher target_threshold results in larger areas being marked as matches for target_colors, which results in more detected black areas in the binary image).
# #     :param ignore_threshold: A high ignore_threshold increases the color range considered similar to ignore_colors, which results in parts of the image ignored (white in the binary image)
# #     :param ignore_colors: Colors to be ignored (set to white) in the processed image.
# #     :return: Processed image.
# #     """
# #     # Start with a white image for the background
# #     print(f'Binarizing Image with {len(target_colors)} target colors...')
# #     binary_image = np.ones_like(img) * 255
# #     # Process target colors
# #     masks = []
# #     for target_color in target_colors:
# #         distance = np.sqrt(np.sum((img - np.array(target_color))**2, axis=2))
# #         mask = distance < target_threshold
# #         masks.append(mask)
# #     print('Masks created succesfully...')
# #     # Combine masks for target colors
# #     combined_mask = np.any(masks, axis=0)
# #     # Exclude the parts of the frame that contain hands
# #     # Set pixels close to any of the target colors to black
# #     binary_image[combined_mask] = [0, 0, 0]
# #     cv.imshow('Binary Image', binary_image)
# #     cv.waitKey(0)
# #     return binary_image

# # def get_contours(img, binary_image, draw_image=True):
# #     lower = np.array([0, 0, 0])
# #     upper = np.array([15, 15, 15])
# #     # the cv.inRange function returns a binary mask, where white pixels (255) represent pixels that are in the range and black pixels (0) pixels that are not.
# #     shapeMask = cv.inRange(binary_image, lower, upper)
# #     (cnts, _) = cv.findContours(shapeMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# #     print('Discarding contours...')
# #     area_thresh = 800
# #     cnts = [c for c in cnts if cv.contourArea(c) > area_thresh]
# #     if draw_image:
# #         for c in cnts:
# #             cv.drawContours(binary_image, [c], -1, (0, 255, 0), 2)
# #             cv.drawContours(img, [c], -1, (0, 255, 0), 10)
# #     return cnts

# # def main():
# #     img_path = 'Pictures/color_splashes.png'
# #     frame = cv.imread(img_path)
# #     cv.imshow('Image', frame)
# #     target_colors = detect_colors(img_path)
# #     # show_colors(target_colors)
# #     binary_image = multi_color_threshold_mask(frame, target_colors, 40)
# #     cnts = get_contours(binary_image, binary_image)
# #     # cnts = get_contours(frame, binary_image)
# #     object_count = len(cnts)
# #     padding = 20
# #     text_x = frame.shape[0]//20
# #     text_y = frame.shape[1]//20
# #     text_size = cv.getTextSize(f'Object Count: {object_count}', cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
# #     cv.rectangle(frame, (text_x - padding, text_y - padding - text_size[1]//2), (text_x + text_size[0] + padding, text_y + text_size[1]//2), (255, 255, 255), cv.FILLED)
# #     cv.putText(frame, f'Object Count: {object_count}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
# #     cv.imshow('Detected Shapes', frame)
# #     cv.waitKey(0)

# # if __name__ == '__main__':
# #     main()

# def detect_colors(img_path):
#     img = cv.imread(img_path)
#     colors = img.reshape(-1, img.shape[-1])
#     unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
#     print(f'Unique Colors: {unique_colors}')
#     background_color = unique_colors[np.argmax(counts)]
#     print(f'Background Color: {background_color}')
#     return unique_colors, background_color

# def show_colors(unique_colors, background_color):
#     # for color in unique_colors:
#     #     print(color)
#     #     color_img = np.full((100, 100, 3), color, dtype='uint8')
#     #     cv.imshow(f'{color}', color_img)
#     #     cv.waitKey(0)
#     background_img = np.full((250, 250, 3), background_color, dtype='uint8')
#     cv.imshow(f'Background Color: {background_color}', background_img)
#     cv.waitKey(0)
    

# if __name__ == '__main__':
#     img_path = 'Pictures/color_splashes.png'
#     img = cv.imread(img_path)
#     cv.imshow('Image', img)
#     unique_colors, background_color = detect_colors(img_path)
#     show_colors(unique_colors, background_color)
#     print('DONE --------------------')
    

# def load_color_dictionary(filename):
#     color_dict = {}
#     with open(filename, 'r') as file:
#         for line in file.readlines():
#             parts = line.split(':')
#             bgr_str = parts[0].strip('()')
#             bgr = tuple(map(int, bgr_str.split(',')))
#             color_name = parts[1].strip()
#             color_dict[bgr] = color_name
#     return color_dict

# def show_color(color_dict):
#     for bgr, color_name in color_dict.items():
#         color_img = np.full((250, 250, 3), bgr, dtype='uint8')
#         cv.imshow(color_name, color_img)
#         cv.waitKey(0)

# if __name__ == '__main__':
#     bgr = (78, 45, 230)
#     color_dict = load_color_dictionary('bgr_color_dictionary.txt')
#     show_color(color_dict)


# def load_color_dictionary(filename):
#     color_dict = {}
#     with open(filename, 'r') as file:
#         for line in file.readlines():
#             parts = line.split(':')
#             bgr_str = parts[0].strip('()')
#             bgr = tuple(map(int, bgr_str.split(',')))
#             color_name = parts[1].strip()
#             color_dict[bgr] = color_name
#     return color_dict

# def show_color(color_dict):
#     for bgr, color_name in color_dict.items():
#         color_img = np.full((250, 250, 3), bgr, dtype='uint8')
#         cv.imshow(color_name, color_img)
#         cv.waitKey(0)

# def categorize_color(bgr, color_dict):
#     # Define the range mapping for each color component
#     ranges = [(0, 63), (63, 126), (126, 189), (189, 252)]
#     # Function to find the closest range
#     closest_bgr = []
#     for c in bgr:
#         for r in ranges:
#             if r[0] <= c and c <= r[1]:
#                 closest_bgr.append(r[0])
#                 break
#     closest_bgr = tuple(closest_bgr)
#     # The color_dict.get() method returns the value for the given key (closest_rgb), or "Not defined" if the key is not found in the dictionary
#     closest_bgr_name = color_dict.get(closest_bgr, "Not defined")
#     return closest_bgr, closest_bgr_name

# if __name__ == '__main__':
#     bgr = (78, 45, 230)
#     color_dict = load_color_dictionary('bgr_color_dictionary.txt')
#     closest_bgr, closest_bgr_name = categorize_color(bgr, color_dict)
#     bgr_img = np.full((250, 250, 3), bgr, dtype='uint8')
#     closest_bgr_img = np.full((250, 250, 3), closest_bgr, dtype='uint8')
#     print(f'The closest color for {bgr} is: {closest_bgr_name}')
#     cv.imshow('BGR', bgr_img)
#     cv.waitKey(0)
#     cv.imshow(f'{closest_bgr}: {closest_bgr_name}', closest_bgr_img)
#     cv.waitKey(0)




def load_color_dict(input_path):
    with open(input_path, 'r') as file:
        color_dict = json.load(file)
    return color_dict

def categorize_color(bgr, color_dict):
    ranges = [(0, 63), (63, 126), (126, 189), (189, 252)]
    closest_bgr = []
    for c in bgr:
        for r in ranges:
            if r[0] <= c <= r[1]:
                closest_bgr.append(r[0])
                break
    closest_bgr = tuple(closest_bgr)
    key_name = str(closest_bgr)
    color_bgr = color_dict[key_name]["BGR"]
    color_name = color_dict[key_name]["Name"]
    print(f'The closest color found is {color_name} with value {color_bgr}')
    return color_name, color_bgr

def show_color(bgr, color_name, color_bgr):
    original_img = np.full((400, 400, 3), bgr, dtype='uint8')
    cv.imshow(f'Passed Color: {bgr}', original_img)
    cv.waitKey(0)
    closest_img = np.full((400, 400, 3), color_bgr, dtype='uint8')
    cv.imshow(f'{color_name}: {color_bgr}', closest_img)
    cv.waitKey(0)
    
if __name__ == '__main__':
    bgr = (142,22,1)
    input_path = 'Dictionaries/bgr_by_key_colors.json'
    color_dict = load_color_dict(input_path)
    color_name, color_bgr = categorize_color(bgr, color_dict)
    show_color(bgr, color_name, color_bgr)