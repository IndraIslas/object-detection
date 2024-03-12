import rembg
import numpy as np
import cv2 as cv

def remove_bg(img):
    img = cv.imread(input_path)
    cv.imshow('Original', img)
    cv.waitKey(0)
    no_bg = rembg.remove(img)
    cv.imshow('No Background', no_bg)
    cv.waitKey(0)
    return no_bg

def brighten_dark_pixels(img, threshold=100, increase_by=80):
    """
    This function brightens pixels in an image that are below a certain threshold (dark pixels).
    :param image_path: Path to the image file.
    :param threshold: Brightness threshold below which pixels will be brightened.
    :param increase_by: The amount by which to increase the brightness of dark pixels.
    :return: The brightened image as a NumPy array.
    """
    # Convert the image to grayscale to determine brightness
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Create a mask where dark pixels are set to True
    dark_pixels_mask = gray_img < threshold
    for i in range(3):
        # Each image in opencv is given as a 3D np array such as np.array([[[], [], ... []], [[], [], ..., []], ..., [[], [], ..., []]])
        # The outermost array represents the image containing arrays that represent each row. 
        # Each row array holds individual pixel arrays (that spread along the columns).
        # Each pixel array contains the three color channel values for that pixel.
        # Therefore, to select a specific pixel, img[h, w, i] selects the h-th row, w-th pixel (column) in that row, and the i-th color channel in that pixel.
        # Increase brightness of the dark pixels. For simplicity, increase the brightness on all BGR channels equally
        # Select all pixels along the height and width dimensions, and pass the boolean mask to select only dark pixels
        img[:, :, i][dark_pixels_mask] += increase_by
    # Ensure that the maximum value for a pixel is 255 with the np.clip function which limits the values of the arrays in img to be between 0 and 255
    img = np.clip(img, 0, 255)
    return img
    
# Find dark pizels and blur the sorounding area
def blur_dark_pixels(img, threshold=100, blur_value=60):
    gray_img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    # Create a mask where dark pixels are set to True
    dark_pixels_mask = gray_img < threshold
    blur = cv.blur(img,(blur_value,blur_value),0)
    out = img.copy()
    out[dark_pixels_mask>0] = blur[dark_pixels_mask>0]
    return out

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

def color_threshold_mask(img, target_color, threshold):
    """
    Process an image to make all colors not close to the target color white, and colors close to the target color black.
    :param img: The input image as a NumPy array (BGR format).
    :param target_color: The BGR color to compare against (as [B, G, R]).
    :param threshold: The distance threshold for color similarity.
    :return: Processed image.
    """
    # Calculate the Euclidean distance for each pixel from the target color
    distance = np.sqrt(np.sum((img - np.array(target_color))**2, axis=2))
    # Create a mask based on the distance threshold
    mask = distance < threshold
    # Create a new image, setting pixels close to the target color to black, others to white
    result_img = np.ones_like(img) * 255  # Start with a white image
    result_img[mask] = [0, 0, 0]  # Set pixels close to the target color to black
    return result_img

def multi_color_threshold_mask(img, target_colors, threshold):
    """
    Process an image to make all colors not close to any target colors white, and colors close to any of the target colors black.
    :param img: The input image as a NumPy array (BGR format).
    :param target_colors: A list of BGR colors to compare against (as [[B, G, R], ...]).
    :param threshold: The distance threshold for color similarity.
    :return: Processed image.
    """
    masks = []
    for target_color in target_colors:
        # Calculate the distance between each pixel value and each target color across all Bgr channels (axis=2) by subtracting and squaring the difference, ensuring all results are positive.
        distance = np.sqrt(np.sum((img - np.array(target_color))**2, axis=2))
        # Create a mask where pixel with distance less than the threshold are set to 1 (True), others to 0 (False)
        mask = distance < threshold
        masks.append(mask)
    # Combine masks so that a pixel close to any of the target colors is set to 1 (True), others to 0 (False)
    combined_mask = np.any(masks, axis=0)
    # Create a new image, setting pixels close to any target color to black, others to white
    result_img = np.ones_like(img) * 255  # Start with a white image
    result_img[combined_mask] = [0, 0, 0]  # Set pixels close to any of the target colors to black
    return result_img

def get_contours(image, original_img, draw_image=True):
    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    shapeMask = cv.inRange(image, lower, upper)
    (cnts, _) = cv.findContours(shapeMask.copy(), cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)
    # print("I found %d black shapes (including false positives)" % len(cnts))
    # cv.imshow("Mask", shapeMask)
    area_thresh = 800
    # print(f"Area Threshold: {area_thresh}")
    # discard contours with area less than area_thresh
    cnts = [c for c in cnts if cv.contourArea(c) > area_thresh]
    # print("I found %d black shapes (excluding false positives)" % len(cnts))
    # cv.destroyAllWindows()
    if draw_image:
        for c in cnts:
            cv.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv.drawContours(original_img, [c], -1, (0, 255, 0), 2)
            # cv.imshow("Processed Image", image)
            # cv.imshow("Original Image", original_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        # get area of contour
        # area = cv.contourArea(c)
        # print(area)
        # cv.imshow("Processed Image", image)
        # cv.imshow("Original Image", original_img)
        # cv.waitKey(0)
    return cnts

def cut_contours(cnts, original_img):
    crops = []
    boxes = []
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # print(x, y, w, h)
        cropped = original_img[y:y+h, x:x+w]
        # cv.imshow("Cropped", cropped)
        # cv.waitKey(0)
        crops.append(cropped)
        boxes.append((x, y, w, h))
    return crops, boxes

def analyze_crops(crops, boxes):
    new_crops = []
    new_boxes = []
    for i,crop in enumerate(crops):
        # get height and width
        h, w, _ = crop.shape
        crop_area = h*w
        area_thresh = 0
        if area_thresh < crop_area:
            new_crops.append(crop)
            new_boxes.append(boxes[i])
        # print(f"crop area: {crop_area}")
        #get median color of crop
        img_temp = crop.copy()
        unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
        img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]
        # print most frequent color 
        # print(unique[np.argmax(counts)])
        # cv.imshow(f"Cropped area {crop_area}", crop)
        # cv.imshow("Most frequent color", img_temp)
        # cv.waitKey(0)
    return new_crops, new_boxes

def get_most_freq_color(crop):
    img_temp = crop.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]
    # print most frequent color 
    print(unique[np.argmax(counts)])

def process_img(img, show=True):
    # Convert to grayscale
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)
    # cv.waitKey(0)
    # Apply Canny Edge Detection
    # brightened = brighten_dark_pixels(img.copy())
    # selective_blur = blur_dark_pixels(img.copy())
    # selective_smooth = smooth_edges_in_dark_areas(img.copy())
    allowed_colors = [[99, 166, 229], [ 95,  80, 202],[170,  98, 222]]
    multiple_color_img = multi_color_threshold_mask(img.copy(), allowed_colors, 40)
    # multiple_color_img = cv.cvtColor(multiple_color_img, cv.COLOR_GRAY2BGR)
    canny = cv.Canny(multiple_color_img, 50, 150)
    # cv.imshow('Canny', bw_img)
    # cv.waitKey(0)
    # Apply Dilation
    dilated = cv.dilate(canny, (20, 20), iterations=2)
    # cv.imshow('Dilated', dilated)
    # cv.waitKey(0)
    # Apply Erosion
    # eroded = cv.erode(dilated, (5, 5),t iterations=1)
    # cv.imshow('Eroded', eroded)
    # cv.waitKey(0)
    # invert colors
    inverted = cv.bitwise_not(dilated)
    bgr_img = cv.cvtColor(inverted, cv.COLOR_GRAY2BGR)
    # cv.imshow('Inverted', bgr_img)
    # cv.waitKey(0)
    if show:
        cv.imshow('1 -> Selective Smooth', multiple_color_img)
        cv.imshow('2 -> Canny', canny)
        cv.imshow('3 -> Dilated', dilated)
        cv.imshow('4 -> Inverted', bgr_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return multiple_color_img

def detect_shapes(img):
    # processed_img = process_img(img)
    allowed_colors = [[99, 166, 229], [ 95, 80, 202],[170, 98, 222],[ 87, 79, 180],[145,  70, 182],[147,  88, 119]]
    multiple_color_img = multi_color_threshold_mask(img.copy(), allowed_colors, 40)
    # processed_img = 
    # draw_image = img.copy()
    cnts = get_contours(multiple_color_img, img)
    # cut out the contours
    # crops, boxes = cut_contours(cnts, img.copy())
    # analyze_crops(crops, boxes)
    # crops, boxes = analyze_crops(crops, boxes)
    # return boxes

def draw_image(img, boxes):
    for box in boxes:
        x, y, w, h = box
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

def main():
    cap = cv.VideoCapture('Videos/video1.mov')
    frame_counter = 0
    boxes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # if frame_counter == 5:
        #     boxes = detect_shapes(frame)
        #     frame_counter = 0
        # draw_image(frame, boxes)
        detect_shapes(frame)
        cv.imshow('frame', frame)
        # frame_counter += 1
        if cv.waitKey(1) == ord('q'):
            break
        for x in range(7):
            cap.read()
    cap.release()
    cv.destroyAllWindows()
    # input_path = 'Images/rectangles.jpeg'

    # original_img = cv.imread(input_path)
    # cv.imshow('Original', original_img)
    # cv.waitKey(0)
    # # img = remove_bg(input_path)
    
    # processed_img = process_img(original_img)
    # draw_image = original_img.copy()
    # cnts = get_contours(processed_img, draw_image)
    # # cut out the contours
    # crops, boxes = cut_contours(cnts, original_img.copy())
    # # analyze_crops(crops, boxes)
    # crops, boxes = analyze_crops(crops, boxes)
    # for box in boxes:
    #     # print(box)
    #     x, y, w, h = box
    #     cv.rectangle(draw_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    # cv.imshow("Original Image", draw_image)
    # cv.waitKey(0)

if __name__ == '__main__':
    main()
    # img = cv.imread('Images/hands.jpeg')
    # boxes = detect_shapes(img)
    # draw_image(img, boxes)
    # cv.imshow('img', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # bright_img = brighten_dark_pixels(img.copy())
    # new_img = blur_dark_pixels(img.copy())
    # smooth_img = smooth_edges_in_dark_areas(img.copy())
    # one_color_img = color_threshold_mask(img.copy(), [99, 166, 229], 30)
    # allowed_colors = [[99, 166, 229], [ 95,  80, 202],[170,  98, 222]]
    # multiple_color_img = multi_color_threshold_mask(img.copy(), allowed_colors, 30)
    # cv.imshow('smooth img', smooth_img)
    # cv.imshow('bright img', bright_img)
    # cv.imshow('multiple color img', multiple_color_img)
    # cv.imshow('one color img', one_color_img)
    # cv.imshow('original img', img)
    # cv.imshow('new img', new_img)
    # cv.waitKey(0)
    # crop = cv.imread('Images/v.png')
    # get_most_freq_color(crop)


# Idea
# Create a mask that selects only the specific color range of the cards that I previously know
# Apply the mask to the image to make the background white
# Apply directly the countour detection to the image