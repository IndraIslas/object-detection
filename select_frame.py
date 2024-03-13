import cv2 as cv
import numpy as np

# Ask marlon if he wants a still frame each time a shape is introduced or once all the shapes are introduced before taking them out
# Set high threshold and compare each still frame with the previous one to see if there was an actual change of there was movement in the video

def resize_frame(frame, scale=0.5):
    new_height = int(frame.shape[0] * scale)
    new_width = int(frame.shape[1] * scale)
    dimensions = (new_width, new_height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def get_still_frames(video_path, threshold=3, similarity_threshold=10):
    """
    Extracts still frames from a video based on the similarity between consecutive frames.
    :param video_path: Path to the video file.
    :param threshold: Difference threshold to consider a frame as still. Lower values means LESS frames will be appended to consecutive stills list.
    :param still_frames: A list of still frames (as numpy arrays).
    :param similarity_threshold: Difference threshold to compare a potential still frame with the last still frame added. Higher values means LESS frames will be appended to still frames list.
    """
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

from PIL import Image
import numpy as np
import cv2 as cv

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
    video_path = 'Videos/video1.mov'
    still_frames = get_still_frames(video_path)
    save_pdf(still_frames, 'still_frames3.pdf')
    print(still_frames)
    print("Returned still frames")
    for i, frame in enumerate(still_frames):
        cv.imshow(f'Still Frame {i}', frame)
    cv.waitKey(0)

if __name__ == '__main__':
    main()