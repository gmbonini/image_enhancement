import os
from argparse import ArgumentParser
import pdb

import tqdm
import cv2
import imutils
import numpy as np

from utils import get_video_files


def get_bounding_boxes(image: np.ndarray, th=5) -> list:

    """
    This functions takes an image with vertical synchronization issue
    and returns a list of bounding boxes for each of its desyncrhonized parts
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image to get the image splits without the black bar
    _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    # Find the contours of the image parts
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    # Extract the bounding boxes for the image parts
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= image.shape[1] * 0.9 and h >= image.shape[0] * 0.025:
            boxes.append(np.array([x, y, x + w, y + h]))
    return boxes


def process_image_parts(boxes: list) -> dict:

    """
    This image process the bounding boxes to extract which coresponds
    to the upper part of the fixed frame and which correponds to the
    lower part of the fixed frame

    boxes = [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2]
    ]
    """

    # If there are two bounding boxes, the image is split into two parts
    if len(boxes) == 2:
        boxes = np.array(boxes)

        # Get the index of the box that has the lowest y1 value (upper part)
        upper_index = np.argmin(boxes[:, 1])

        # Get the index of the box that has the highest y1 value (lower part)
        lower_index = np.argmax(boxes[:, 1])

        parts = {"upper": boxes[upper_index], "lower": boxes[lower_index]}

    # If there's only one bounding box, the image is not split but there's a
    # black bar at the bottom of the image
    elif len(boxes) == 1:
        parts = {"upper": boxes}

    # There's no black bar and no synchronization issue
    else:
        return None

    return parts


def stitch_image_parts(parts: dict, image: np.ndarray) -> np.ndarray:

    """
    This function is responsible for stitching the parts of an image together.
    The lower part should be above the upper part, because the vertical sync issue
    is causing the video to "move" vertically from the bottom to the top,
    so the upper part of the next frame shows up at the bottom
    """

    # The image has only a black bar at the bottom
    if "lower" not in parts.keys():
        upper_box = parts["upper"][0]

        # Crop out the black bar
        upper_image = image[upper_box[1] : upper_box[3], upper_box[0] : upper_box[2]]

        stitched = upper_image

    else:
        upper_box = parts["upper"]
        lower_box = parts["lower"]

        # Crop each part of the image
        upper_image = image[upper_box[1] : upper_box[3], upper_box[0] : upper_box[2]]
        lower_image = image[lower_box[1] : lower_box[3], lower_box[0] : lower_box[2]]

        # Get the width of the smaller crop
        min_width = min(upper_image.shape[1], lower_image.shape[1])

        # Resize both crops so that they can be stacked together
        upper_image = cv2.resize(upper_image, (min_width, upper_image.shape[0]))
        lower_image = cv2.resize(lower_image, (min_width, lower_image.shape[0]))

        # Stack the lower part above the upper part to complete the frame
        stitched = np.vstack((lower_image, upper_image))

    return stitched


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Script responsible to stitch together frames of video with vertical synchronization issues"
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        type=str,
        help="Path to the directory containing the video files",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path to the output directory, where the fixed videos will be saved",
    )
    args = parser.parse_args()

    # Get the video path and the output directory from the CLI arguments
    video_dir = args.video_dir
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    video_files = get_video_files(video_dir)
    for video_path in video_files:

        # The output video name will be the name of the original video followed by '_fixed'
        output_video_name = video_path.split("/")[-1].replace(".mp4", "_fixed.mp4")
        output_path = os.path.join(output_dir, output_video_name)

        # Open the input video and get it's shape and FPS
        cap = cv2.VideoCapture(video_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create the output video writer using the same properties of the input video
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        # Initialize the frame count to show a progress bar
        index = 0
        # Initialize the progress bar
        progress_bar = tqdm.tqdm(total=frame_count)

        while cap.isOpened():

            # pdb.set_trace()

            # Read the video frame
            ret, frame = cap.read()
            if not ret:
                break

            # Update the progress_bar
            progress_bar.update(1)

            # Get the bounding boxes for each part of the image
            boxes = get_bounding_boxes(frame)

            # Get which part of the image should be on top of the other
            parts = process_image_parts(boxes)

            # Initialize the output frame
            stitched = frame

            # If the algorithm detected that the video frame is desyncrhonized,
            # stitch the image parts into a single frame
            if parts is not None:
                stitched = stitch_image_parts(parts, frame)
                stitched = cv2.resize(stitched, (frame.shape[1], frame.shape[0]))

            # Write the stitched frame to the output video
            writer.write(stitched)
            index += 1

        # Release memory
        cap.release()
        writer.release()
        print(
            f"Stitching completed for video {video_path.split('/')[-1]}. Result saved to {output_path}"
        )
