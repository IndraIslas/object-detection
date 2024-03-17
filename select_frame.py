import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp

def resize_frame(frame, scale=0.5):
    new_height = int(frame.shape[0] * scale)
    new_width = int(frame.shape[1] * scale)
    dimensions = (new_width, new_height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

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
        detected_image = cv.flip(frame.copy(), 1)
        hands_detected = False
        hand_landmarks_list = []
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
                hand_landmarks_list.append(landmarks)
                # hand_landmarks is a list of all landmarks of a single hand where each landmark is a tuple with the x and y coordinates of that part of the hand.
                landmarks = [(int(landmark.x * frame_width), int(landmark.y * frame_height)) for landmark in hand_landmarks.landmark]
                hull = cv.convexHull(np.array(landmarks, dtype=np.int32))
                cv.drawContours(detected_image, [hull], -1, (0, 255, 0), 2)
            return detected_image, hands_detected, hand_landmarks_list
        else:
            print("No hands were found")
            return detected_image, hands_detected, hand_landmarks_list

def get_still_frames(video_path, threshold=3, similarity_threshold=10, include_hands=False):
    """
    Extracts still frames from a video based on the similarity between consecutive frames.
    :param video_path: Path to the video file.
    :param threshold: Difference threshold to consider a frame as still. Lower values means LESS frames will be appended to consecutive stills list.
    :param still_frames: A list of still frames (as numpy arrays).
    :param similarity_threshold: Difference threshold to compare a potential still frame with the last still frame added. Higher values means LESS frames will be appended to still frames list.
    """
    print(video_path, threshold, similarity_threshold, include_hands)
    cap = cv.VideoCapture(video_path)
    prev_canny_frame = None
    still_frames = []
    consecutive_stills = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = resize_frame(frame, 0.25)
        canny_frame = cv.Canny(resized_frame, 100, 200)
        dilated_canny = cv.dilate(canny_frame, (5, 5), iterations=3)
        # cv.imshow('Current frame', frame)
        if prev_canny_frame is not None:
            frame_diff_score = np.mean(cv.absdiff(prev_canny_frame, dilated_canny))
            print(frame_diff_score)
            if frame_diff_score < threshold:
                consecutive_stills.append(frame)
                print("Frame added to consecutive stills")
            else:
                if len(consecutive_stills) > 1:
                    middle_frame = consecutive_stills[len(consecutive_stills) // 2]
                    resized_middle_frame = resize_frame(middle_frame, 0.25)
                    canny_middle_frame = cv.Canny(resized_middle_frame, 100, 200)
                    dilated_canny_middle = cv.dilate(canny_middle_frame, (5, 5), iterations=3)
                    append = True
                    for i in range(1, min(len(still_frames), 5)):
                        # The lower the similarity threshold is, the more frames will be appended to still frames
                        still_middle_diff_score = np.mean(cv.absdiff(dilated_canny_last_still, dilated_canny_middle))
                        if still_frames and (still_middle_diff_score < similarity_threshold):
                            append = False
                    if append:
                        if include_hands:
                            print("New still frame found")
                            still_frames.append(middle_frame)
                            dilated_canny_last_still = dilated_canny_middle
                        else:
                            _, hands_detected, _ = detect_hands(middle_frame)
                            if not hands_detected:
                                print("New still frame found")
                                still_frames.append(middle_frame)
                                dilated_canny_last_still = dilated_canny_middle
                    else:
                        print("Similar to last frame in still frames --- Discarded")
                else:
                        print("Movement detected --- Discarded")
                consecutive_stills = []
        prev_canny_frame = dilated_canny
        for i in range(7):
            cap.read()
    cap.release()
    return still_frames

def save_pdf(still_frames, save_path):
    """
    Saves the given frames as a single PDF file.
    :param still_frames: A list of still frames (as numpy arrays).
    :param save_path: The path to save the PDF file to.
    """
    # Convert NumPy arrays to PIL images
    pil_images = [Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) for frame in still_frames]
    # Optionally resize images for consistency and to reduce PDF size
    resized_images = []
    for img in pil_images:
        img.thumbnail((1500, 1000), Image.ANTIALIAS)
        resized_images.append(img)
    # Save the first image and append the rest to a PDF file
    if resized_images:
        resized_images[0].save(save_path, "PDF", resolution=100.0, save_all=True, append_images=resized_images[1:])

def main():
    video_path = 'Videos/video2.mov'
    include_hands=False
    still_frames = get_still_frames(video_path, threshold=10, similarity_threshold=25, include_hands=include_hands)
    save_pdf(still_frames, f'still_frames_{include_hands}_3.pdf')
    # print(still_frames)
    print("PDF SAVED --------------------")
    # for i, frame in enumerate(still_frames):
    #     cv.imshow(f'Still Frame {i}', frame)
    # cv.waitKey(0)

if __name__ == '__main__':
    main()