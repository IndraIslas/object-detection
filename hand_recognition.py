import cv2 as cv
import mediapipe as mp
import math
import numpy as np

# IDEAS: make this a function and return: 
# a list of tuples with the coordinates of the detected hands;
# GET_CONTOURS: I think the best thing would be running the get_contours function after the detect_hands function so that the get_contours function ignores the hands on the frame and only selects the rest of contours.
# FINAL RESULT: The final result should be a video/live video that shows the detected hands and the contours of the objects with an object counter and saves all the still frames to a pdf.
# PDF: For the pdf, since this is going to run on live video, maybe make the function so that it saves the pdf every 10 frames or so, and then appends the following still frames to the same pdf (overwritting)

# def resize_and_show(image, desired_height=480, desired_width=480):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv.resize(image, (desired_width, math.floor(h/(w/desired_width))))
#   else:
#     img = cv.resize(image, (math.floor(w/(h/desired_height)), desired_height))
#   cv.imshow('Resized',img)
#   cv.waitKey(0)

# def detect_hands_video(video_path):
#     video = cv.VideoCapture(video_path)
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles
#     # Run MediaPipe Hands.
#     with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 print("Can't receive frame (stream end?). Exiting...")
#                 break
#             # Convert the BGR frame to RGB and flip the frame around y-axis for correct handedness output 
#             flipped_frame_rgb = cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2RGB), 1)
#             # Process the frame with MediaPipe Hands.
#             results = hands.process(flipped_frame_rgb)
#             # Draw hand landmarks of each hand.
#             frame_height, frame_width, _ = frame.shape
#             detected_image = cv.flip(frame, 1)
#             hands_detected = False
#             hand_landmarks_list = []
#             if results.multi_hand_landmarks:
#                 hands_detected = True
#                 # Print handedness (left v.s. right hand).
#                 print(f'Handedness: {results.multi_handedness}')
#                 # Select a single hand (hand_landmarks) from the list of all hands (results.multi_hand_landmarks)
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Print index finger tip coordinates.
#                     print(f'Index finger tip coordinate: (',f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width}, 'f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height})')
#                     mp_drawing.draw_landmarks(
#                         detected_image,
#                         hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style())
#                     # hand_landmarks is a class of Mediapipe of all landmarks of a single hand where each landmark is a tuple with the x and y coordinates of that part of the hand.
#                     landmarks = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in hand_landmarks.landmark]
#                     # The hand_landmarks_list is a list of lists where the inner list contain the landmarks of a single hand as tuples with the x and y coordinates of that part of the hand.
#                     hand_landmarks_list.append(landmarks)
#                     # hand_landmarks is a list of all landmarks of a single hand where each landmark is a tuple with the x and y coordinates of that part of the hand.
#                     landmarks = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in hand_landmarks.landmark]
#                     hull = cv.convexHull(np.array(landmarks, dtype=np.int32))
#                     cv.drawContours(detected_image, [hull], -1, (0, 255, 0), 2)
#                 yield hands_detected, hand_landmarks_list
#             else:
#                 print("No hands were found")
#                 yield hands_detected, hand_landmarks_list
#             cv.imshow("Detected Hands Video", (cv.flip(detected_image, 1)))
#             if cv.waitKey(20) & 0xFF==ord('d'):
#                 break
#         cv.destroyAllWindows()
#         video.release()

# for hands_detected, hand_landmarks_list in detect_hands_video('Videos/Video1.mov'):
#     pass

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
        detected_image = cv.flip(frame, 1)
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
                    detected_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                # hand_landmarks is a class of Mediapipe of all landmarks of a single hand where each landmark is a tuple with the x and y coordinates of that part of the hand.
                landmarks = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in hand_landmarks.landmark]
                # The hand_landmarks_list is a list of lists where the inner list contain the landmarks of a single hand as tuples with the x and y coordinates of that part of the hand.
                hands_landmarks_list.append(landmarks)
                hull = cv.convexHull(np.array(landmarks, dtype=np.int32))
                cv.drawContours(detected_image, [hull], -1, (0, 255, 0), 2)
            return detected_image, hands_detected, hands_landmarks_list
        else:
            print("No hands were found")
            return detected_image, hands_detected, hands_landmarks_list

def generate_hands_mask(frame, hands_landmarks_list):
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    # For each hand detected, create a convex hull and fill it on the mask.
    for hand_landmarks in hands_landmarks_list:
        hull = cv.convexHull(np.array(hand_landmarks, dtype=np.int32))
        cv.drawContours(mask, [hull], -1, 255, -1)
    cv.imshow("Masked", mask)
    cv.waitKey(0)
    return mask
    # # Invert the mask so that the hands are zeros and the rest is 255
    # inverted_mask = cv.bitwise_not(mask)
    # # Create a purple frame of the same size as the original frame
    # purple_frame = np.zeros_like(frame)
    # purple_frame[:, :] = [128, 0, 128]  # BGR for purple
    # # Apply the inverted mask to the purple frame
    # purple_only = np.where()
    # # Apply the original mask to the original frame (to keep the hand regions unchanged)
    # hands_only = cv.bitwise_and(frame, frame, mask=mask)
    # # Combine the two frames
    # final_frame = cv.add(purple_only, hands_only)
    # return final_frame

# def main():
#     video_path = 'Videos/Video1.mov'
#     video = cv.VideoCapture(video_path)
#     while video.isOpened():
#         ret, frame = video.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting...")
#             break
#         detected_frame, _, _ = detect_hands(frame)
#         detected_frame = cv.flip(detected_frame, 1)
#         cv.imshow('Detected Hands', detected_frame)
#         if cv.waitKey(20) & 0xFF==ord('d'):
#             break
#     cv.destroyAllWindows()
#     video.release()

if __name__ == '__main__':
    # main()
    img = cv.imread('Pictures/twohands.png')
    cv.imshow('Original', img)



# import cv2
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#         results = hands.process(image)

#         if results.multi_hand_landmarks:
#             image_height, image_width, _ = image.shape
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())
#         else:
#             print("no hands were found")
                
#         cv2.imshow("result", image)
#         cv2.waitKey(0)