import cv2 as cv
import math
import numpy as np
import mediapipe as mp

def resize_frame(frame, desired_width=480, desired_height=400):
    h, w = frame.shape[:2]
    if h < w:
        frame = cv.resize(frame, (desired_width, math.floor(h/(w/desired_width))))
    else:
        img = cv.resize(frame, (math.floor(w/(h/desired_height)), desired_height))
        cv.imshow('Resized', img)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
help(mp_hands.Hands)

# Run MediaPipe Hands.
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7) as hands:
    # Convert the BGR image to RGB, flip the image around y-axis for correct 
    # handedness output and process it with MediaPipe Hands.
    results = hands.process(cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2RGB), 1))

    # Print handedness (left v.s. right hand).
    print(results.multi_handedness)
    if not results.multi_hand_landmarks:
      print('No hand landmarks found in this frame.')
    else:
        # Draw hand landmarks of each hand.
        image_hight, image_width, _ = frame.shape
        annotated_image = cv.flip(frame.copy(), 1)
        for hand_landmarks in results.multi_hand_landmarks:
        # Print index finger tip coordinates.
            print(f'Index finger tip coordinate: ({hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, {hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})')
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            resize_show(cv.flip(annotated_image, 1))



# Reading and displaying video
img = cv.imread('Pictures/hands.png')
