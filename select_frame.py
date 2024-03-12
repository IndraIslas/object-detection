import cv2 as cv
import numpy as np

# Ask marlon if he wants a still frame each time a shape is introduced or once all the shapes are introduced before taking them out
# Set high threshold and compare each still frame with the previous one to see if there was an actual change of there was movement in the video

def get_still_frames(video_path, threshold=0.8):
    """
    Extracts still frames from a video based on the similarity between consecutive frames.
    :param video_path: Path to the video file.
    :param threshold: Difference threshold to consider a frame as still. Lower values mean more similarity is required.
    :param still_frames: A list of still frames (as numpy arrays).
    """
    cap = cv.VideoCapture(video_path)
    prev_blurred_frame = None
    still_frames = []
    consecutive_stills = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred_frame = cv.GaussianBlur(frame, (15, 15), cv.BORDER_DEFAULT)
        # cv.imshow('Current frame', frame)
        if prev_blurred_frame is not None:
            frame_diff = cv.absdiff(prev_blurred_frame, blurred_frame)
            frame_diff_score = np.mean(frame_diff)
            print(frame_diff_score)
            if frame_diff_score < threshold:
                consecutive_stills.append(frame)
                print("Frame added to consecutive stills")
            else:
                if len(consecutive_stills) > 1:
                    middle_frame = consecutive_stills[len(consecutive_stills) // 2]
                    blurred_middle_frame = cv.GaussianBlur(middle_frame, (15, 15), cv.BORDER_DEFAULT)
                    append = True
                    for i in range(1, min(len(still_frames), 5)):
                        if still_frames and not (np.mean(cv.absdiff(cv.GaussianBlur(still_frames[-i], (5, 5), cv.BORDER_DEFAULT), blurred_middle_frame)) >= threshold):
                            append = False
                    if append:
                        print("New still frame found")
                        still_frames.append(middle_frame)
                    else:
                        print("Similar to last frame in still frames --- Discarded")
                else:
                        print("Movement detected --- Discarded")
                consecutive_stills = []
        prev_blurred_frame = blurred_frame
        for i in range(10):
            cap.read()
    cap.release()
    return still_frames

def main():
    video_path = 'Videos/video1.mov'
    still_frames = get_still_frames(video_path)
    print(still_frames)
    print("Returned still frames")
    for i, frame in enumerate(still_frames):
        cv.imshow(f'Still Frame {i}', frame)
    cv.waitKey(0)

if __name__ == '__main__':
    main()