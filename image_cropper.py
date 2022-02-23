import os
import cv2
import numpy as np
import imutils
import rawpy

from argparse import ArgumentParser
from glob import glob
from imutils.paths import list_images
from tqdm import tqdm

sp = os.path.sep


class ImageCropper:
    def __init__(self, directory: str, output_dir: str, overcrop: float) -> None:

        self.directory = directory
        self.output_dir = output_dir
        self.overcrop = overcrop

        # List all images that are saved in a compressed format (JPEG, PNG, etc)
        compressed_images_list = list(list_images(directory))

        # List all images saved in a raw format with capitalized characters (CR3)
        raw_images_list = list(glob(os.path.join(self.directory, "*.CR3")))

        # List all images saved in a raw format with lowercase characters (cr3)
        raw_images_list_lower = list(glob(os.path.join(self.directory, "*.cr3")))

        # Sum all the image lists
        self.images_list = (
            set(compressed_images_list + raw_images_list + raw_images_list_lower)
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
        if len(boxes) == 1:
            box = boxes[0]
        elif len(boxes) == 0:
            box = None
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

        # First, try the dark approach
        box = self.dark_approach(image)

        # If the dark approach resulted in a bounding box that encloses almost
        # all the image, switch to the bright approach
        if box is None:
            box = self.bright_approach(image)
        elif (
            box[2] - box[0] > 0.865 * image.shape[1]
            or box[3] - box[1] > 0.865 * image.shape[0]
        ):
            box = self.bright_approach(image)

        # Crop the image based on the bounding box values
        cropped = image[box[1] : box[3], box[0] : box[2]]

        increase_crop_x = int(cropped.shape[1] * self.overcrop)
        increase_crop_y = int(cropped.shape[0] * self.overcrop)
        cropped = cropped[
            increase_crop_y : cropped.shape[0] - increase_crop_y,
            increase_crop_x : cropped.shape[1] - increase_crop_x,
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

            image_name = image_path.split(sp)[-1].split(".")[0]
            image_name = image_name.replace(" ","_")
            image_name = image_name.replace("'","_")
            image_name = image_name.replace(",","_")
            
            new_path = os.path.join(self.output_dir, image_name + ".jpg")

            # Save the new image
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
    parser.add_argument(
        "-c",
        "--overcrop",
        type=float,
        default=5,
        help="Increase the crop based on percentage value (the overcrop between 0 and 100), defaults to 5",
    )

    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    overcrop = args.overcrop / 100

    os.makedirs(output, exist_ok=True)

    croper = ImageCropper(directory, output, overcrop)
    croper.crop()
