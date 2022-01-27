import os
import pdb
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
from imutils.paths import list_images
import imutils
from tqdm import tqdm

import rawpy


class ImageCropper:
    def __init__(self, directory: str, output_dir: str) -> None:

        self.directory = directory
        self.output_dir = output_dir

        # List all images that are saved in a compressed format (JPEG, PNG, etc)
        compressed_images_list = list(list_images(directory))

        # List all images saved in a raw format with capitalized characters (CR3)
        raw_images_list = list(glob(os.path.join(self.directory, "*.CR3")))

        # List all images saved in a raw format with lowercase characters (cr3)
        raw_images_list_lower = list(glob(os.path.join(self.directory, "*.cr3")))

        # Sum all the image lists
        self.images_list = (
            compressed_images_list + raw_images_list + raw_images_list_lower
        )

    def bright_approach(self, image, th=10):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly bright
        """

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize the image to get the image splits without the black bar
        _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

        # Find the contours of the binary image
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # Get the biggest contour (should be the picture contour)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(contour)

        box = np.array([x, y, x + w, y + h])

        return box

    def dark_approach(self, image):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly dark
        """

        # Convert the image to the HSV color space
        res = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(res)

        # Find the contours of the saturation channel
        contours = cv2.findContours(hue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # Get the bounding boxes for the most significant contours
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > image.shape[1] * 0.1 and h > image.shape[0] * 0.1:
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
                boxes.append(np.array([x, y, x + w, y + h]))
        boxes = np.array(boxes, dtype=np.int32)

        # Get the bounding box that encloses all the bounding boxes
        if len(boxes) == 0:
            box = boxes[0]
        else:
            x1 = np.min(boxes[:, 0])
            y1 = np.min(boxes[:, 1])
            x2 = np.max(boxes[:, 2])
            y2 = np.max(boxes[:, 3])
            box = np.array([x1, y1, x2, y2], dtype=np.int32)

        return box

    def remove_borders(self, image: np.ndarray, draw=True) -> np.ndarray:

        """
        This functions takes an image, binarize it to get what is an image
        and what is just a black border, get the image contours and crop out
        the black borders
        """

        eps_x = int(image.shape[1] * (0.5 / 100))
        eps_y = int(image.shape[0] * (0.5 / 100))

        # First, try the dark approach
        box = self.dark_approach(image)

        # If the dark approach resulted in a bounding box that encloses almost
        # all the image, switch to the bright approach
        if (
            box[2] - box[0] > 0.865 * image.shape[1]
            or box[3] - box[1] > 0.865 * image.shape[0]
        ):
            box = self.bright_approach(image)

        # Crop the image based on the bounding box values
        cropped = image[
            box[1] + eps_y : box[3] - eps_y, box[0] + eps_x : box[2] - eps_x
        ]

        return cropped

    def crop(self):
        for image_path in tqdm(self.images_list):

            # Convert raw images (CR3 files) to numpy arrays
            if image_path.lower().endswith(".cr3"):
                with rawpy.imread(image_path) as raw:
                    image = raw.postprocess()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Read compressed images with OpenCV
            else:
                image = cv2.imread(image_path, 1)

            # Remove the black borders
            cropped = self.remove_borders(image)

            # Save the new image
            new_path = os.path.join(
                self.output_dir, image_path.split("/")[-1].split(".")[0] + ".jpg"
            )
            cv2.imwrite(new_path, cropped)

        print(f"Cropped pictures saved to {self.output_dir}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        help="Path to the directory containing the image files",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path to the output directory, where the cropped images will be saved",
    )

    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    os.makedirs(output, exist_ok=True)

    croper = ImageCropper(directory, output)
    croper.crop()
