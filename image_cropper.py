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
        self.default_overcrop = overcrop
        self.overcrop_black_box = self.default_overcrop
        self.overcrop_white_box = self.default_overcrop * 0.95
        self.overcrop = overcrop

        # List all images that are saved in a compressed format (JPEG, PNG, etc)
        compressed_images_list = list(list_images(directory))

        # List all images saved in a raw format with capitalized characters (CR3)
        raw_images_list = list(glob(os.path.join(self.directory, "*.CR3")))

        # List all images saved in a raw format with lowercase characters (cr3)
        raw_images_list_lower = list(glob(os.path.join(self.directory, "*.cr3")))

        # Sum all the image lists
        self.images_list = set(
            compressed_images_list + raw_images_list + raw_images_list_lower
        )

    def bright_approach(self, image, th=10):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly bright
        """

        self.overcrop_black_box = self.default_overcrop * 0.6

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize the image to get the image splits without the black bar
        _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

        dilate = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        erosion = cv2.erode(
            dilate, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
        )
        open = cv2.morphologyEx(
            erosion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        )

        # Find the contours of the binary image
        contours = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # Get the bounding boxes for the most significant contours
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > image.shape[1] * 0.1 and h > image.shape[0] * 0.1:
                # cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
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

    def dark_approach(self, image):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly dark
        """

        self.overcrop_black_box = self.default_overcrop

        # Convert the image to the HSV color space
        res = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(res)

        erosion = cv2.erode(
            hue, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1
        )

        open = cv2.morphologyEx(
            erosion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )

        # Find the contours of the saturation channel
        contours = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # Get the bounding boxes for the most significant contours
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > image.shape[1] * 0.1 and h > image.shape[0] * 0.1:
                # cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
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

    def box_verification(self, black_box, white_box, image):

        """
        This functions compare the size of the bounding boxes and returns the
        the appropriate bounding box to use for crop.
        """

        black_box_height = black_box[3] - black_box[1]
        black_box_width = black_box[2] - black_box[0]
        white_box_height = white_box[3] - white_box[1]
        white_box_width = white_box[2] - white_box[0]
        image_height = image.shape[0]

        black_box_area = black_box_height * black_box_width
        white_box_area = white_box_height * white_box_width

        if white_box_area >= black_box_area * 0.90 and white_box_area <= black_box_area:
            self.overcrop = self.overcrop_white_box
            return white_box
        elif black_box_height == image_height:
            self.overcrop = self.overcrop_white_box
            return white_box
        elif black_box[3] <= image_height and black_box[3] >= image_height - 10:
            self.overcrop = self.overcrop_white_box
            return white_box
        elif black_box[1] <= 10:
            self.overcrop = self.overcrop_white_box
            return white_box
        elif white_box_width >= black_box_width:
            if white_box_height >= black_box_height:
                self.overcrop = self.overcrop_white_box
                return white_box
            else:
                self.overcrop = self.overcrop_black_box
                return black_box
        else:
            self.overcrop = self.overcrop_black_box
            return black_box

    def remove_borders(
        self, image: np.ndarray, draw_both=False, draw_final=False
    ) -> np.ndarray:

        """
        This functions takes an image, binarize it to get what is an image
        and what is just a black border, get the image contours and crop out
        the black borders
        """

        original_image = image.copy()
        image = cv2.resize(image, (1280, 720))

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

        white_box = self.white_borders(image)

        if draw_both == True:
            cv2.rectangle(
                image,
                (
                    int(box[0]),
                    int(box[1]),
                ),
                (
                    int(box[2]),
                    int(box[3]),
                ),
                (0, 0, 255),
                1,
            )

            cv2.rectangle(
                image,
                (
                    int(white_box[0]),
                    int(white_box[1]),
                ),
                (
                    int(white_box[2]),
                    int(white_box[3]),
                ),
                (0, 255, 0),
                1,
            )

            cv2.namedWindow("Both Methods", 0)
            cv2.imshow("Both Methods", image)
            k = cv2.waitKey(0)
            if k == ord("q"):
                exit()

        # Compare the detected bounding boxes of both methods and returns the appropriate box.
        box = self.box_verification(box, white_box, image)

        normalized_box = []
        normalized_box.append(box[0] / image.shape[1])
        normalized_box.append(box[1] / image.shape[0])
        normalized_box.append(box[2] / image.shape[1])
        normalized_box.append(box[3] / image.shape[0])

        if draw_final == True:
            cv2.rectangle(
                original_image,
                (
                    int(normalized_box[0] * original_image.shape[1]),
                    int(normalized_box[1] * original_image.shape[0]),
                ),
                (
                    int(normalized_box[2] * original_image.shape[1]),
                    int(normalized_box[3] * original_image.shape[0]),
                ),
                (0, 0, 255),
                5,
            )

            cv2.namedWindow("Result", 0)
            cv2.imshow("Result", original_image)
            k = cv2.waitKey(0)
            if k == ord("q"):
                exit()

        # Crop the image based on the bounding box values
        cropped = original_image[
            int(normalized_box[1] * original_image.shape[0]) : int(
                normalized_box[3] * original_image.shape[0]
            ),
            int(normalized_box[0] * original_image.shape[1]) : int(
                normalized_box[2] * original_image.shape[1]
            ),
        ]

        increase_crop_x = int(cropped.shape[1] * self.overcrop)
        increase_crop_y = int(cropped.shape[0] * self.overcrop * 1.15)

        cropped = cropped[
            increase_crop_y : cropped.shape[0] - increase_crop_y,
            increase_crop_x : cropped.shape[1] - increase_crop_x,
        ]

        return cropped

    def white_borders(self, image):

        img = cv2.Canny(image, 80, 150)
        dilate = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        cnts = cv2.findContours(
            image=dilate, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )

        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnt = cnts[0]

        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h])

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

        return box

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

            image_name = image_path.split(sp)[-1].split(".")[0]
            image_name = image_name.replace(" ", "_")
            image_name = image_name.replace("'", "_")
            image_name = image_name.replace(",", "_")

            # Remove the black borders
            cropped = self.remove_borders(image)

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
        default=1.0857,
        help="Increase the crop based on percentage value (the overcrop between 0 and 100), defaults to 5",
    )

    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    overcrop = args.overcrop / 100

    os.makedirs(output, exist_ok=True)

    croper = ImageCropper(directory, output, overcrop)
    croper.crop()
