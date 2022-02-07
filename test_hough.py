import os
import pdb
from argparse import ArgumentParser
from glob import glob
import math

import cv2
import numpy as np
from imutils.paths import list_images
import imutils
from tqdm import tqdm

import rawpy


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
            compressed_images_list + raw_images_list + raw_images_list_lower
        )

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
        if (
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
            # cropped = self.remove_borders(image)
            self.hough(image)

            # Save the new image
            # new_path = os.path.join(
            #     self.output_dir, image_path.split("/")[-1].split(".")[0] + ".jpg"
            # )
            # cv2.imwrite(new_path, cropped)

        print(f"Cropped pictures saved to {self.output_dir}")

    def hough(self, image: np.ndarray) -> np.ndarray:

        image = cv2.GaussianBlur(image, (5, 5), 0)
        res = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(res)

        image = imutils.adjust_brightness_contrast(
            image, brightness=30.0, contrast=50.0
        )

        cv2.namedWindow(
            "t", cv2.WINDOW_NORMAL
        )  # Create window with freedom of dimensions

        # canny = imutils.auto_canny(image, 0.33)
        contours = cv2.findContours(sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        boxes = []
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (255, 0, 255), 4)
        #     boxes.append(cv2.boundingRect(contour))

        # boxes = np.array(boxes, dtype=np.int32)
        # x1 = np.min(boxes[:, 0])
        # x2 = np.min(boxes[:, 0])
        # x3 = np.min(boxes[:, 0])
        # x1 = np.min(boxes[:, 0])

        # lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)

        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         cv2.line(image, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

        # linesP = cv2.HoughLinesP(canny, 1, np.pi / 6, 10, None, 0, 0)

        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv2.line(
        #             image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 10, cv2.LINE_AA
        #         )

        cv2.imshow("t", image)
        k = cv2.waitKey(0)
        if k == ord("q"):
            exit()

        # cv2.imshow("t", image)
        # k = cv2.waitKey(0)
        # if k == ord("q"):
        #     exit()
        return 0


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
