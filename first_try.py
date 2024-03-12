from PIL import Image, ImageEnhance
import rembg
import numpy as np
import cv2 as cv

input_path = '/Users/indraislas/Desktop/Mine/ll/firstproject/Pictures/hands.png'
img = cv.imread(input_path)

def remove_background(input_path):
    # input_path = '/Users/indraislas/Desktop/Mine/ll/firstproject/Pictures/Multiple.png'
    #read image and show it
    img = cv.imread(input_path)
    cv.imshow('Original', img)
    cv.waitKey(0)
    no_bg = rembg.remove(img)
    cv.imshow('No Background', no_bg)
    cv.waitKey(0)
    # no_bg_path = '/Users/indraislas/Desktop/Mine/ll/firstproject/Pictures/no_bgs.png'
    # cv.imwrite(no_bg_path, no_bg)
    return no_bg

def black_shapes(input_path):
    # # image = remove_background(input_path)
    # no_bg_path = remove_background(input_path)
    # image = Image.open(no_bg_path)
    # # Enhance contrast
    # enhancer = ImageEnhance.Contrast(image)
    # factor = 5  # Increase contrast
    # enhanced_image = enhancer.enhance(factor)
    # # Convert PIL Image to NumPy array
    # enhanced_image_np = np.array(enhanced_image)
    # # Convert RGB to BGR for OpenCV
    # enhanced_image_np = cv.cvtColor(enhanced_image_np, cv.COLOR_RGB2BGR)
    # # Display image
    # cv.imshow('Enhanced Image', enhanced_image_np)
    # cv.waitKey(0)
    no_bg = remove_background(input_path)
    canny = cv.Canny(no_bg, threshold1=125, threshold2=175)
    cv.imshow('Canny', canny)
    cv.waitKey(0)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # thick_contours = cv.drawContours(no_bg.copy(), contours, -1, (0, 0, 0), thickness=20)
    dilated = cv.dilate(canny, (20, 20), iterations=3)
    cv.imshow('Dilated', dilated)
    cv.waitKey(0)
    # cv.imshow('Thick Contours', thick_contours)
    # cv.waitKey(0)
    # gray = cv.cvtColor(canny, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)
    # cv.waitKey(0)
    # Converting the image to binary
    # retval, binary = cv.threshold(canny, thresh=105, maxval=255, type=cv.THRESH_BINARY)
    # cv.imshow('Binary', binary)
    # cv.waitKey(0)
    bitwise_not = cv.bitwise_not(dilated)
    cv.imshow('Bitwise Not', bitwise_not)
    cv.waitKey(0)
    bitwise_not_bgr = cv.cvtColor(bitwise_not, cv.COLOR_GRAY2BGR)
    return bitwise_not_bgr

def recognize_shapes(input_path):
    blackwhite = black_shapes(input_path)
    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    shapeMask = cv.inRange(blackwhite, lower, upper)
    (cnts, _) = cv.findContours(shapeMask.copy(), cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)
    print("I found %d black shapes" % len(cnts))
    cv.imshow("Mask", shapeMask)
    cv.waitKey(0)
    for c in cnts:
        cv.drawContours(blackwhite, [c], -1, (0, 255, 0), 2)
        cv.imshow("Image", blackwhite)
        cv.waitKey(0)
    recgonized_shapes = cv.drawContours(img, cnts, -1, (0, 255, 0), 2)
    cv.imshow('Recognized Shapes', recgonized_shapes)
    cv.waitKey(0)

if __name__ == '__main__':
    input_path = '/Users/indraislas/Desktop/Mine/ll/firstproject/Pictures/hands.png'
    recognize_shapes(input_path)
