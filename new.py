import cv2 as cv
import mediapipe as mp
import numpy as np

# Train the color as soon as it appears on screen

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
                cv.drawContours(hands_image, [hull], -1, (0, 255, 0), 2)
            return hands_image, hands_detected, hands_landmarks_list
        else:
            print("No hands were found")
            return hands_image, hands_detected, hands_landmarks_list

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

def multi_color_threshold_mask(img, target_colors, threshold, hand_mask):
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
    # Exclude the parts of the frame that contain hands
    combined_mask[hand_mask == 255] = False
    # Create a new image, setting pixels close to any target color to black, others to white
    binary_image = np.ones_like(img) * 255  # Start with a white image
    binary_image[combined_mask] = [0, 0, 0]  # Set pixels close to any of the target colors to black
    # cv.imshow('Binary Image', binary_image)
    # cv.waitKey(0)
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

# def max_objects():
#     max_objects = input('Enter the maximum number of cards used: ')
#     max_objects = int(max_objects)
#     return max_objects

def main():
    video = cv.VideoCapture('Videos/Video1.mov')
    # cards_total = max_objects()
    # save_counter = 0
    while video.isOpened():
        isTrue, frame = video.read()
        if not isTrue:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # hands_image, _, hands_landmarks_list = detect_hands(frame)
        _, _, hands_landmarks_list = detect_hands(frame)
        # hands_image = cv.flip(hands_image, 1)
        hand_mask = generate_hands_mask(frame, hands_landmarks_list)
        target_colors = [[99, 166, 229], [ 95, 80, 202],[170, 98, 222],[ 87, 79, 180],[145,  70, 182],[147,  88, 119]]
        binary_image = multi_color_threshold_mask(frame, target_colors, 40, hand_mask)
        # cnts = get_contours(hands_image, binary_image)
        cnts = get_contours(frame, binary_image)
        object_count = len(cnts)
        padding = 20
        # text_x = hands_image.shape[0]//20
        # text_y = hands_image.shape[1]//20
        # text_size = cv.getTextSize(f'Object Count: {object_count}', cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        # cv.rectangle(hands_image, (text_x - padding, text_y - padding - text_size[1]//2), (text_x + text_size[0] + padding, text_y + text_size[1]//2), (255, 255, 255), cv.FILLED)
        # cv.putText(hands_image, f'Object Count: {object_count}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        # cv.imshow('Detected Shapes', hands_image)
        text_x = frame.shape[0]//20
        text_y = frame.shape[1]//20
        text_size = cv.getTextSize(f'Object Count: {object_count}', cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv.rectangle(frame, (text_x - padding, text_y - padding - text_size[1]//2), (text_x + text_size[0] + padding, text_y + text_size[1]//2), (255, 255, 255), cv.FILLED)
        cv.putText(frame, f'Object Count: {object_count}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow('Detected Shapes', frame)
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
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
