from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
from PIL import Image
import mediapipe as mp
from collections import Counter
from sklearn.cluster import KMeans

prev_canny_frame = None
still_frames = []
consecutive_stills = []
include_hands=False
dilated_canny_last_still = None

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

def get_still_frames(frame, threshold, similarity_threshold, include_hands=False):
    """
    Extracts still frames from a video based on the similarity between consecutive frames.
    :param video_path: Path to the video file.
    :param threshold: Difference threshold to consider a frame as still. LOWER values means LESS frames will be appended to consecutive stills list.
    :param similarity_threshold: Difference threshold to compare a potential still frame with the last still frame added. HIGHER values means LESS frames will be appended to still frames list.
    :param include_hands: Whether to include frames with hands in the still frames list.
    :return still_frames: A list of still frames (as numpy arrays).
    """
    global prev_canny_frame, still_frames, consecutive_stills, dilated_canny_last_still
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
                    if still_frames:
                        # The lower the similarity threshold is, the more frames will be appended to still frames
                        still_middle_diff_score = np.mean(cv.absdiff(dilated_canny_last_still, dilated_canny_middle))
                        if still_middle_diff_score < similarity_threshold:
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

# def find_background_and_target_colors(still_frames, hand_landmarks_list):
#     background_colors = []
#     all_colors = []
#     for index, frame in enumerate(still_frames):
#         rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         # Flatten the frame array and reduce colors such that the frame has as many rows as necessary (-1) to have 3 columns (representing the RGB channels)
#         reshaped_frame = rgb_frame.reshape((-1, 3))
#         # Initialize an instance kmeans of the class KMeans, which implements the K-Means clustering algorithm, which is used for partitioning n observations into k clusters in which each observation belongs to the cluster with the nearest mean,
#         kmeans = KMeans(n_clusters=1)
#         # The method .fit takes an pixel data in 2D array (reshaped_frame) and assigns each data point to one of the n_clusters defined during the initialization of the KMeans instance
#         kmeans.fit(reshaped_frame)
#         # The attribute cluster_centers_ (after the 2D array has been fitted into kmeans) stores the coordinates of the cluster centers that the algorithm has identified.
#         most_frequent_color = kmeans.cluster_centers_[0]
#         # Append the most frequent color to the background_colors list by rounding the RGB values and converting them to integers, and the array to a tuple to make it more manageable and 
#         background_colors.append(tuple(np.round(most_frequent_color).astype(int)))
#         if hand_landmarks_list:
#             mask = np.zeros(frame.shape[:2], dtype="uint8")
#             for hand_landmarks in hand_landmarks_list[index]:
#                 # The fillPolly function fills an area bounded by several polygonal contours with a specific color (white) or intensity. It takes the image to be filled, the polygonal contours, and the color as arguments.
#                 cv.fillPoly(mask, [np.array(hand_landmarks, dtype=np.int32)], (255))
#             # Inverts the mask such that the white hand hulls become black and the rest white
#             masked_frame = cv.bitwise_and(rgb_frame, rgb_frame, mask=cv.bitwise_not(mask))
#             # Select the colors in the rows of masked_frame that are not black (sum of the RGB values is greater than 0) and convert them to a set to remove duplicates
#             unique_colors = set(tuple(color) for row in masked_frame for color in row if sum(color) > 0)
#             all_colors.extend(unique_colors)
#     # The class Counter maps elements to their counts such that it counts how many times each color appears in the background-color list
#     # The method most_common(n) returns a n tuples with the most common elements and their respective counts from the most common to the least. [0][0] accesses the first element of the list returned by most_common(1).
#     if background_colors:
#         most_common_bg_color = Counter(background_colors).most_common(1)[0][0]
#         # Exclude the background color and convert to a set for uniqueness since all_colors can contain duplicates from different frames even if unique_colors contains different colors per frame.
#         target_colors = set(all_colors) - {most_common_bg_color}
#         return list(target_colors), most_common_bg_color
#     return set(), None

def find_background_and_target_colors(frame, hand_landmarks_list):
    frame_colors = []
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Flatten the frame array and reduce colors such that the frame has as many rows as necessary (-1) to have 3 columns (representing the RGB channels)
    reshaped_frame = rgb_frame.reshape((-1, 3))
    # Initialize an instance kmeans of the class KMeans, which implements the K-Means clustering algorithm, which is used for partitioning n observations into k clusters in which each observation belongs to the cluster with the nearest mean,
    kmeans = KMeans(n_clusters=1)
    # The method .fit takes an pixel data in 2D array (reshaped_frame) and assigns each data point to one of the n_clusters defined during the initialization of the KMeans instance
    kmeans.fit(reshaped_frame)
    # The attribute cluster_centers_ (after the 2D array has been fitted into kmeans) stores the coordinates of the cluster centers that the algorithm has identified.
    most_frequent_color = kmeans.cluster_centers_[0]
    # Append the most frequent color to the background_colors list by rounding the RGB values and converting them to integers, and the array to a tuple to make it more manageable and 
    background_color = tuple(np.round(most_frequent_color).astype(int))
    if hand_landmarks_list:
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for hand_landmarks in hand_landmarks_list:
            # The fillPolly function fills an area bounded by several polygonal contours with a specific color (white) or intensity. It takes the image to be filled, the polygonal contours, and the color as arguments.
            cv.fillPoly(mask, [np.array(hand_landmarks, dtype=np.int32)], (255))
        # Inverts the mask such that the white hand hulls become black and the rest white
        masked_frame = cv.bitwise_and(rgb_frame, rgb_frame, mask=cv.bitwise_not(mask))
        # Select the colors in the rows of masked_frame that are not black (sum of the RGB values is greater than 0) and convert them to a set to remove duplicates
        frame_colors = set(tuple(color) for row in masked_frame for color in row if sum(color) > 0)
    else:
        frame_colors = set(tuple(color) for row in rgb_frame for color in row)
    return frame_colors, background_color

def merge_colors(all_colors, n_clusters):
    """
    Merge similar colors using K-Means clustering.
    :param colors: A list of RGB colors (tuples).
    :param n_clusters: The number of clusters to form. This effectively determines the number of merged colors you'll end up with.
    : returns merged_colors: A list of merged colors (tuples), one for each cluster.
    """
    # Convert the list of colors to a numpy array for processing
    color_arr = np.array(all_colors)
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(color_arr)
    # Calculate the centroid of each cluster to represent the merged color
    centroids = kmeans.cluster_centers_
    # Convert centroids to tuples of integers (RGB values should be integers)
    merged_colors = [tuple(map(int, centroid)) for centroid in centroids]
    return merged_colors

def filter_bg(merged_colors, background_colors, n_clusters):
    """
    Remove colors similar to the background color using K-Means clustering.
    :param all_colors: A list of RGB color tuples detected in still_frames.
    :param background_colors: A list of RGB tuple representing the background color of each still frame.
    :param n_clusters: Number of clusters to use for K-Means. This number should be set based on the diversity of colors and the inclusion of the background color.
    :returns filtered_colors: A list of colors, excluding those similar to the background color.
    """
    if background_colors:
        # The class Counter maps elements to their counts such that it counts how many times each color appears in the background-color list
        # The method most_common(n) returns a n tuples with the most common elements and their respective counts from the most common to the least. [0][0] accesses the first element of the list returned by most_common(1).
        background_color = Counter(background_colors).most_common(1)[0][0]
        # Exclude the background color and convert to a set for uniqueness since all_colors can contain duplicates from different frames even if unique_colors contains different colors per frame.
        colors_for_clustering = np.array(merged_colors + [background_color])
        # Create an instance of the KMeans class, kmeans, with n_clusters clusters and fit the colors_for_clustering array to these clusters.
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(colors_for_clustering)
        # The cluster centers are the average of the points in each cluster and are stored in the cluster_centers_ attribute of the kmeans instance.
        centroids = kmeans.cluster_centers_
        # The .predict() method assigns each data point (each color) to the nearest cluster centroid. 
        # Since background_color is a single color value, .predict will return a single index, write [0] to access the first element instead of handling an array with a single value.
        background_cluster_index = kmeans.predict([background_color])[0]
        filtered_colors = []
        for color, label in zip(colors_for_clustering, kmeans.labels_):
            # Filter out colors that are in the same cluster as the background color.
            if label != background_cluster_index and tuple(color) != tuple(background_color):
                # The map function applies the int function to each element of the tuple color so that all floating points are converted to integers; the tuple function then converts the integer list to a tuple
                filtered_colors.append(list(map(int, color)))
        return filtered_colors, background_color
    return set(merged_colors), None

# def get_final_background_and_target_colors(merged_colors, background_colors):
#     if background_colors:
#         # The class Counter maps elements to their counts such that it counts how many times each color appears in the background-color list
#         # The method most_common(n) returns a n tuples with the most common elements and their respective counts from the most common to the least. [0][0] accesses the first element of the list returned by most_common(1).
#         most_common_bg_color = Counter(background_colors).most_common(1)[0][0]
#         # Exclude the background color and convert to a set for uniqueness since all_colors can contain duplicates from different frames even if unique_colors contains different colors per frame.
#         target_colors = set(merged_colors) - {most_common_bg_color}
#         print(target_colors)
#         return set(target_colors), most_common_bg_color
#     return set(merged_colors), None

def main(): # Do the cropping function 75%
    video_path = 'Videos/video1.mov'
    cap = cv.VideoCapture(video_path)
    include_hands = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        get_still_frames(frame, threshold=3, similarity_threshold=13, include_hands=include_hands)
        cv.imshow('Current frame', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
        for i in range(20):
            cap.read()
    cap.release()
    cv.destroyAllWindows()
    save_pdf(still_frames, f'PDFs/still-frames-{include_hands}.pdf')
    print('STILL FRAMES SAVED AS PDF --------------------')
    all_colors = []
    background_colors = []
    for frame in still_frames:
        _, _, hand_landmarks_list = detect_hands(frame)
        frame_colors, background_color = find_background_and_target_colors(frame, hand_landmarks_list)
        all_colors.extend(frame_colors)
        background_colors.append(background_color)
    if background_colors:
        for color in background_colors:
            target_image = np.full((100, 100, 3), color, dtype=np.uint8)
            cv.imshow('Target Color', target_image)
            cv.waitKey(0)
    merged_colors = merge_colors(all_colors, n_clusters=50)
    target_colors, background_color = filter_bg(merged_colors, background_colors, n_clusters=20)
    print(target_colors)
    with open('Colors/target-colors.txt', 'w') as file:
        file.write(str(target_colors))
    with open('Colors/background-colors.txt', 'w') as file:
        file.write(str(background_color))
    print('TARGET COLORS SAVED AS TXT --------------------')
    print('BACKGROUND COLOR SAVED AS TXT --------------------')
    if target_colors:
        for color in target_colors:
            target_image = np.full((100, 100, 3), color, dtype=np.uint8)
            cv.imshow('Target Color', target_image)
            cv.waitKey(0)
    if background_color:
        bg_image = np.full((500, 500, 3), background_color, dtype=np.uint8)
        cv.imshow('Background Color', bg_image)
        cv.waitKey(0)
    print('TRAINING COMPLETED --------------------')

if __name__ == '__main__':
    main()