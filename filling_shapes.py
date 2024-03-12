import cv2 as cv
import numpy as np

def smooth_edges_in_dark_areas(img, darkness_threshold=100, smoothing_kernel_size=(9, 9)):
    """
    Smoothes the edges of very dark areas in an image.
    (Smoothes the intersecting areas between the dilated canny image and the inverse2 binary image whose pixel values are 255 (white))
    :param img: Input image as a NumPy array (read by OpenCV).
    :param darkness_threshold: Brightness threshold to identify dark pixels.
    :param smoothing_kernel_size: Size of the Gaussian kernel used for smoothing edges.
    :return: Image with smoothed edges in dark areas.
    """
    # Convert the image to grayscale to determine brightness
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Getting the inverse binary image
    _, binary = cv.threshold(gray_img, darkness_threshold, 255, cv.THRESH_BINARY_INV)
    cv.imshow('Dark Areas Mask', binary)
    cv.waitKey(0)
    # Detect edges in the dark areas
    canny_binary = cv.Canny(binary, 100, 200)
    cv.imshow('Dark Areas Edges', canny_binary)
    cv.waitKey(0)
    # Dilate edges to ensure more comprehensive smoothing
    dilated_edges = cv.dilate(canny_binary, np.ones((3, 3), np.uint8), iterations=3)
    cv.imshow('Dilated Edges', dilated_edges)
    cv.waitKey(0)
    # Create a mask where edges are dilated
    edge_mask = cv.bitwise_and(binary, dilated_edges)
    cv.imshow('Edge Mask', edge_mask)
    cv.waitKey(0)
    # Apply Gaussian blur to smooth the edges in the original image
    smoothed_img = cv.GaussianBlur(img, smoothing_kernel_size, 3.0)
    # np.where(condition, x, y) function selects the elements in a np array such that if they meet the condition (True) their value is set to x, otherwise it is set to y
    final_image = np.where(edge_mask[:,:,None] == 255, smoothed_img, img)
    return final_image


img = cv.imread('Pictures/23.png')
img2 = smooth_edges_in_dark_areas(img, 100, (9, 9))
cv.imshow('Original', img)
cv.imshow('Smoothed', img2)
cv.waitKey(0)