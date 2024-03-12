import cv2 as cv
import numpy as np

# Train the color as soon as it appears on screen

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
    area_thresh = 800
    cnts = [c for c in cnts if cv.contourArea(c) > area_thresh]
    if draw_image:
        for c in cnts:
            cv.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv.drawContours(original_img, [c], -1, (0, 255, 0), 10)
    return cnts

# def get_contours(image, original_img, draw_image=True):
#     lower = np.array([0, 0, 0])
#     upper = np.array([15, 15, 15])
#     shapeMask = cv.inRange(image, lower, upper)
#     cnts, _ = cv.findContours(shapeMask.copy(), cv.RETR_EXTERNAL,
#                                     cv.CHAIN_APPROX_SIMPLE)
#     area_thresh = 800
#     cnts = [c for c in cnts if cv.contourArea(c) > area_thresh]
#     if draw_image:
#         for c in cnts:
#             # Approximate the contour
#             epsilon = 0.001 * cv.arcLength(c, True)
#             approx = cv.approxPolyDP(c, epsilon, True)
#             # Draw the approximated contour
#             cv.drawContours(image, [approx], -1, (0, 255, 0), 2)
#             cv.drawContours(original_img, [approx], -1, (0, 255, 0), 10)
#     return cnts

def detect_shapes(img):
    allowed_colors = [[99, 166, 229], [ 95, 80, 202],[170, 98, 222],[ 87, 79, 180],[145,  70, 182],[147,  88, 119]]
    multiple_color_img = multi_color_threshold_mask(img.copy(), allowed_colors, 40)
    cnts = get_contours(multiple_color_img, img)
    return len(cnts)

def max_objects():
    max_objects = input('Enter the maximum number of cards used: ')
    max_objects = int(max_objects)
    return max_objects

def main():
    video = cv.VideoCapture('Videos/Video1.mov')
    cards_total = max_objects()
    save_counter = 0
    while video.isOpened():
        isTrue, frame = video.read()
        if not isTrue:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        object_count = detect_shapes(frame)
        padding = 20
        text_x = frame.shape[0]//20
        text_y = frame.shape[1]//20
        text_size = cv.getTextSize(f'Object Count: {object_count}', cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv.rectangle(frame, (text_x - padding, text_y - padding - text_size[1]//2), (text_x + text_size[0] + padding, text_y + text_size[1]//2), (255, 255, 255), cv.FILLED)
        cv.putText(frame, f'Object Count: {object_count}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow('Detected Shapes', frame)
        if 0 < object_count and object_count <= cards_total:
            save_path = f'saved_frames/frame_{save_counter}.png'
            cv.imwrite(save_path, frame)
            print('Saved succesfully')
            save_counter += 1
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
        for i in range(7):
            video.read()
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
