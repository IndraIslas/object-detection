import cv2 as cv
import numpy as np
import cv2

# def black_shapes(image_path):
#     try:
#         img = cv.imread(image_path)
#         cv.imshow('Original', img)
#         canny = cv.Canny(img, threshold1=10, threshold2=255)
#         cv.imshow('Canny', canny)
#         contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         # Filter out contours based on their area
#         filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 5000]
#         print(f"Number of filtered contours: {len(filtered_contours)}")
#         # Draw filled contours
#         filled_contours = cv.drawContours(canny.copy(), filtered_contours, -1, (255, 0, 0), thickness=cv.FILLED)
#         bitwise_not = cv.bitwise_not(filled_contours)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#     return bitwise_not

# image_path = '/Users/indraislas/Desktop/Mine/ll/firstproject/Pictures/Multiple.png'
image = cv.imread('/Users/indraislas/Desktop/Mine/ll/firstproject/Pictures/blackandwhite.png')
cv.imshow('Original', image)

lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv.inRange(image, lower, upper)

(cnts, _) = cv.findContours(shapeMask.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
print("I found %d black shapes" % len(cnts))
cv.imshow("Mask", shapeMask)

for c in cnts:
    cv.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv.imshow("Image", image)

cv.waitKey(0)
