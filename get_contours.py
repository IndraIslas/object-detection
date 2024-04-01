from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from collections import Counter
from sklearn.cluster import KMeans

# Train the color as soon as it appears on screen
# Idea: get the most frequent color of the frame (the table) and then setting it to white
# Idea: getting the most frequent color inside the hull and then setting it to white
# Don't merge all files into a single script, rather call the scripts so that they are separated by functionility

def detect_hands(frame):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    # Run MediaPipe Hands.
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        # Convert the BGR frame to RGB and flip the frame around y-axis for correct handedness output
        flipped_frame_rgb = cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2RGB), 1)
        # Process the frame with MediaPipe Hands.
        results = hands.process(flipped_frame_rgb)
        # Draw hand landmarks of each hand.
        frame_height, frame_width, _ = frame.shape
        hands_image = cv.flip(frame, 1)
        hands_detected = False
        hands_landmarks_list = []
        if results.multi_hand_landmarks:
            hands_detected = True
            # Print handedness (left v.s. right hand).
            print(f'Handedness: {results.multi_handedness}')
            # Select a single hand (hand_landmarks) from the list of all hands (results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                # Print index finger tip coordinates.
                print(f'Index finger tip coordinate: (',f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width}, 'f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height})')
                mp_drawing.draw_landmarks(
                    hands_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                # hand_landmarks is a class of Mediapipe of all landmarks of a single hand where each landmark is a tuple with the x and y coordinates of that part of the hand.
                landmarks = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in hand_landmarks.landmark]
                # The hand_landmarks_list is a list of lists where the inner list contain the landmarks of a single hand as tuples with the x and y coordinates of that part of the hand.
                hands_landmarks_list.append(landmarks)
                hull = cv.convexHull(np.array(landmarks, dtype=np.int32))
                # cv.drawContours(hands_image, [hull], -1, (0, 255, 0), 2)
            return hands_image, hands_detected, hands_landmarks_list
        else:
            print("No hands were found")
            return hands_image, hands_detected, hands_landmarks_list

def crop_and_mask_image(frame, hands_landmarks_list):
    crops = []
    frame_width = frame.shape[1]  # Get the width of the frame
    for hand_landmarks in hands_landmarks_list:
        # Convert landmarks to hull and then adjust for the original frame's orientation
        hull = cv.convexHull(np.array(hand_landmarks, dtype=np.int32)) 
        # Adjust hull points to match the original frame's orientation since the current hull is flipped
        corrected_hull = []
        for point in hull:
            corrected_x = frame_width - point[0][0]  # Mirror the x coordinate
            corrected_hull.append([[corrected_x, point[0][1]]])
        corrected_hull = np.array(corrected_hull, dtype=np.int32)
        # Now corrected_hull is aligned with the original frame, proceed as before
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [corrected_hull], -1, (255, 255, 255), -1)
        # Use the corrected hull for cropping
        x, y, w, h = cv.boundingRect(corrected_hull)
        cropped_frame = frame[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]
        # Applying the mask to the cropped frame
        cropped_frame_masked = cv.bitwise_and(cropped_frame, cropped_frame, mask=mask_cropped)
        cropped_frame_masked = cv.flip(cropped_frame_masked, 1)
        blurred_cropped = cv.GaussianBlur(cropped_frame_masked, (5, 5), 0)
        crops.append(blurred_cropped)
        cv.imshow('Blurred Cropped', blurred_cropped)
        # crops.append(cropped_frame_masked)
        # cv.imshow('Cropped Frame', cropped_frame_masked)
        cv.waitKey(0)
    return crops

def get_most_freq_color(crop):
    img_temp = crop.copy()
    # The img_temp.reshape(-1, 3) operation transforms the image into a two-dimensional array of shape [n, 3], where n (rows) is the number of pixels and 3 (columns) is the three color channel values , effectively converting the image data into a color list and removing spatial structure.
    # The parameter axis=0 tells np.unique to treat each row (each pixel color value) as a single entity and to find unique rows.
    # The np.unique function in NumPy identifies unique elements in an array, and optionally, with parameters like axis and return_counts, it can find unique rows or columns and return the frequencies of these unique values or arrays, respectively.
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]
    # print most frequent color 
    print(unique[np.argmax(counts)])

def generate_hands_mask(frame, hands_landmarks_list):
    hand_mask = np.zeros(frame.shape[:2], dtype="uint8")
    # For each hand detected, create a convex hull and fill it on the mask.
    for hand_landmarks in hands_landmarks_list:
        hull = cv.convexHull(np.array(hand_landmarks, dtype=np.int32))
        cv.drawContours(hand_mask, [hull], -1, 255, -1)
    hand_mask = cv.flip(hand_mask, 1)
    # cv.imshow("Hand Mask", hand_mask)
    # cv.waitKey(0)
    return hand_mask

def multi_color_threshold_mask(img, target_colors, target_threshold, ignore_threshold, hand_mask, ignore_colors=[]):
    """
    Process an image to make all colors not close to any target colors white, and colors close to any of the target colors black.
    Pixels close to ignore_colors are set to white.
    :param img: The input image as a NumPy array (BGR format).
    :param target_colors: A list of BGR colors to compare against (as [[B, G, R], ...]).
    :param target_threshold: A higher target_threshold results in larger areas being marked as matches for target_colors, which results in more detected black areas in the binary image).
    :param ignore_threshold: A high ignore_threshold increases the color range considered similar to ignore_colors, which results in parts of the image ignored (white in the binary image)
    :param ignore_colors: Colors to be ignored (set to white) in the processed image.
    :return: Processed image.
    """
    # Start with a white image for the background
    binary_image = np.ones_like(img) * 255
    # Process target colors
    masks = []
    for target_color in target_colors:
        distance = np.sqrt(np.sum((img - np.array(target_color))**2, axis=2))
        mask = distance < target_threshold
        masks.append(mask)
    # Combine masks for target colors
    combined_mask = np.any(masks, axis=0)
    # Exclude the parts of the frame that contain hands
    # Set pixels close to any of the target colors to black
    combined_mask[hand_mask == 255] = False
    binary_image[combined_mask] = [0, 0, 0]
    if ignore_colors:
        # Process ignore_colors first
        for ignore_color in ignore_colors:
            distance = np.sqrt(np.sum((img - np.array(ignore_color))**2, axis=2))
            ignore_mask = distance < ignore_threshold
            # For pixels close to ignore_color, keep them white in binary_image
            binary_image[ignore_mask] = [255, 255, 255]
            cv.imshow('Ignore Mask', binary_image)
    cv.imshow('Binary Image', binary_image)
    return binary_image

def get_contours(img, binary_image, draw_image=True):
    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    shapeMask = cv.inRange(binary_image, lower, upper)
    (cnts, _) = cv.findContours(shapeMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area_thresh = 800
    cnts = [c for c in cnts if cv.contourArea(c) > area_thresh]
    if draw_image:
        for c in cnts:
            cv.drawContours(binary_image, [c], -1, (0, 255, 0), 2)
            cv.drawContours(img, [c], -1, (0, 255, 0), 10)
    return cnts

def parse_color_set(target_colors_path):
    with open(target_colors_path, 'r') as file:
        content = file.read()
        color_set = eval(content)
        color_list = list(color_set)
    return color_list

# def merge_colors(colors, n_clusters=25):
#     """
#     Merge similar colors using K-Means clustering.
#     :param colors: A list of RGB colors (tuples).
#     :param n_clusters: The number of clusters to form. This effectively determines the number of merged colors you'll end up with.
#     : returns merged_colors: A list of merged colors (tuples), one for each cluster.
#     """
#     # Convert the list of colors to a numpy array for processing
#     color_arr = np.array(colors)
#     # Apply K-Means clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(color_arr)
#     # Calculate the centroid of each cluster to represent the merged color
#     centroids = kmeans.cluster_centers_
#     # Convert centroids to tuples of integers (RGB values should be integers)
#     merged_colors = [tuple(map(int, centroid)) for centroid in centroids]
#     return merged_colors

def main():
    video = cv.VideoCapture(0)
    color_set_path = 'Colors/target-colors.txt'
    isTrue, frame = video.read()  # Read once to get the frame size
    if not isTrue:
        print("Failed to read video")
        return
    frame_height, frame_width = frame.shape[:2]
    # FourCC is a four-character code used to specify the video codec in media file formats
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', 'X264', depending on the desired output format
    # Specify the output file name, fourcc code (codec), frames per second (FPS), and frame size. The frame size should match your input video's frame size.
    out = cv.VideoWriter('output_video2.mp4', fourcc, 20.0, (frame_width, frame_height))
    # Rewind the video to its beginning by setting the next frame to be the first frame.
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    while video.isOpened():
        isTrue, frame = video.read()
        if not isTrue:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        hands_image, _, hands_landmarks_list = detect_hands(frame)
        # _, _, hands_landmarks_list = detect_hands(frame)
        hands_image = cv.flip(hands_image, 1)
        hand_mask = generate_hands_mask(frame, hands_landmarks_list)
        # target_colors = [[99, 166, 229], [ 95, 80, 202],[170, 98, 222],[ 87, 79, 180],[145,  70, 182],[147,  88, 119], [237,120,120], [93,164,234], [62,214,214], [96,181,224], [239,162,200]]
        target_colors = [[64, 126, 160], [128, 106, 122], [110, 145, 145], [140, 133, 98]]
        # target_colors = parse_color_set(color_set_path)
        # target_colors = merge_colors(color_set, n_clusters=10)
        binary_image = multi_color_threshold_mask(frame, target_colors, 30, 30, hand_mask, ignore_colors=[[119, 121, 116], [255,255,255]])
        # binary_image = multi_color_threshold_mask(frame, target_colors, 40)
        cnts = get_contours(hands_image, binary_image)
        # cnts = get_contours(frame, binary_image)
        object_count = len(cnts)
        padding = 20
        text_x = hands_image.shape[0]//20
        text_y = hands_image.shape[1]//20
        text_size = cv.getTextSize(f'Object Count: {object_count}', cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv.rectangle(hands_image, (text_x - padding, text_y - padding - text_size[1]//2), (text_x + text_size[0] + padding, text_y + text_size[1]//2), (255, 255, 255), cv.FILLED)
        cv.putText(hands_image, f'Object Count: {object_count}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        out.write(hands_image)
        cv.imshow('Detected Shapes', hands_image)
        # text_x = frame.shape[0]//20
        # text_y = frame.shape[1]//20
        # text_size = cv.getTextSize(f'Object Count: {object_count}', cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # cv.rectangle(frame, (text_x - padding, text_y - padding - text_size[1]//2), (text_x + text_size[0] + padding, text_y + text_size[1]//2), (255, 255, 255), cv.FILLED)
        # cv.putText(frame, f'Object Count: {object_count}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        # cv.imshow('Detected Shapes', frame)
        # if 0 < object_count and object_count <= cards_total:
        #     save_path = f'saved_frames/frame_{save_counter}.png'
        #     cv.imwrite(save_path, frame)
        #     print('Saved succesfully')
        #     save_counter += 1
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
        for i in range(15):
            video.read()
    video.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
