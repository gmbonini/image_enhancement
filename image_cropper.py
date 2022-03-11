import os
import cv2
import numpy as np
import imutils
import rawpy

from argparse import ArgumentParser
from glob import glob
from imutils.paths import list_images
from tqdm import tqdm
from skimage.filters import threshold_niblack
from skimage.restoration import denoise_wavelet

sp = os.path.sep


class ImageCropper:
    def __init__(
        self, directory: str, output_dir: str, overcrop: float, white_border
    ) -> None:

        self.directory = directory
        self.output_dir = output_dir
        self.default_overcrop = overcrop
        self.overcrop_black_box = self.default_overcrop
        self.overcrop_white_box = self.default_overcrop * 0.95
        self.overcrop = overcrop
        self.white_border_method = white_border
        self.reading_raw_images = True

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
                # cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
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
            hue, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
        )
        open = cv2.morphologyEx(
            erosion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 6))
        )
        dilate = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        if self.reading_raw_images:
            # Find the contours of the dilatation
            contours = cv2.findContours(
                dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = imutils.grab_contours(contours)
        else:
            # Find the contours of the hue channel
            contours = cv2.findContours(hue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)

        # Get the bounding boxes for the most significant contours
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > image.shape[1] * 0.1 and h > image.shape[0] * 0.1:
                # cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
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

        if self.reading_raw_images:

            resized_image = cv2.resize(image, (1280, 720))

            normalized_box = []
            normalized_box.append(box[0] / image.shape[1])
            normalized_box.append(box[1] / image.shape[0])
            normalized_box.append(box[2] / image.shape[1])
            normalized_box.append(box[3] / image.shape[0])

            normalized_box[0] = int(normalized_box[0] * resized_image.shape[1])
            normalized_box[1] = int(normalized_box[1] * resized_image.shape[0])
            normalized_box[2] = int(normalized_box[2] * resized_image.shape[1])
            normalized_box[3] = int(normalized_box[3] * resized_image.shape[0])

            return normalized_box
        else:
            return box

    def crop_analyzer(self, crop):

        image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if not self.reading_raw_images:
            window_size = 75
            k = 0.5
            points_treshold = 100

            # Niblack threshold.
            thresh_niblack = threshold_niblack(image, window_size=window_size, k=k)
            binary_niblack = image > thresh_niblack
            binary_niblack = np.ascontiguousarray(binary_niblack, dtype=np.float32)
            binary = binary_niblack * 255
            binary = cv2.bitwise_not(binary.astype(np.uint8))

            pts = np.argwhere(binary != 0)

            # If most of the crop is black: return True.
            if pts.shape[0] < points_treshold:
                return True
            else:
                return False

        else:
            window_size = 105
            k = 0.8
            points_treshold = 100
            # points_treshold = 2000

            # Niblack threshold.
            thresh_niblack = threshold_niblack(image, window_size=window_size, k=k)
            binary_niblack = image > thresh_niblack
            binary_niblack = np.ascontiguousarray(binary_niblack, dtype=np.float32)
            binary = binary_niblack * 255
            binary = cv2.bitwise_not(binary.astype(np.uint8))
            pts = np.argwhere(binary != 0)

            # print(f"Number of points: {pts.shape[0]}")
            
            # If most of the crop is black: return True.
            if pts.shape[0] < points_treshold:
                return True
            else:
                return False

    def first_box_verification(self, black_box, white_box, threshold=30):

        """
        This functions compare the size of the bounding boxes and returns the
        the appropriate bounding box to use for crop.
        """

        if black_box is None:
            return white_box

        if white_box is None:
            return black_box

        P1_black = [black_box[0], black_box[1]]
        P4_black = [black_box[2], black_box[3]]

        P1_white = [white_box[0], white_box[1]]
        P4_white = [white_box[2], white_box[3]]

        # P1 - X
        if (abs(P1_black[0] - P1_white[0])) <= threshold:
            p1_x = max(P1_black[0], P1_white[0])
        else:
            p1_x = min(P1_black[0], P1_white[0])
        # P1 - Y
        if (abs(P1_black[1] - P1_white[1])) <= threshold:
            p1_y = max(P1_black[1], P1_white[1])
        else:
            p1_y = min(P1_black[1], P1_white[1])

        # P4 - X
        if (abs(P4_black[0] - P4_white[0])) <= threshold:
            p4_x = min(P4_black[0], P4_white[0])
        else:
            p4_x = max(P4_black[0], P4_white[0])
        # P4 - Y
        if (abs(P4_black[1] - P4_white[1])) <= threshold:
            p4_y = min(P4_black[1], P4_white[1])
        else:
            p4_y = max(P4_black[1], P4_white[1])

        box = np.array([p1_x, p1_y, p4_x, p4_y], dtype=np.int32)

        return box

    def box_verification(self, black_box, white_box, image, threshold=30):

        """
        This functions compare the size of the bounding boxes and returns the
        the appropriate bounding box to use for crop.
        """

        if black_box is None:
            return white_box

        if white_box is None:
            return black_box

        P1_black = [black_box[0], black_box[1]]
        P2_black = [black_box[2], black_box[1]]
        P3_black = [black_box[0], black_box[3]]
        P4_black = [black_box[2], black_box[3]]

        P1_white = [white_box[0], white_box[1]]
        P2_white = [white_box[2], white_box[1]]
        P3_white = [white_box[0], white_box[3]]
        P4_white = [white_box[2], white_box[3]]

        # P1 - X
        if (abs(P1_black[0] - P1_white[0])) <= threshold:
            p1_x = max(P1_black[0], P1_white[0])
        # Analyze the left crop.
        else:
            p1x = min(P1_black[0], P1_white[0])
            p1y = min(P1_black[1], P1_white[1])
            p3x = max(P3_black[0], P3_white[0])
            p3y = max(P3_black[1], P3_white[1])
            crop_left = image[p1y:p3y, p1x:p3x]
            if self.crop_analyzer(crop_left):
                # If most of the crop is black:
                p1_x = max(P1_black[0], P1_white[0])
            else:
                p1_x = min(P1_black[0], P1_white[0])
        # P1 - Y
        if (abs(P1_black[1] - P1_white[1])) <= threshold:
            p1_y = max(P1_black[1], P1_white[1])
        # Analyze the top crop.
        else:
            p1x = min(P1_black[0], P1_white[0])
            p1y = min(P1_black[1], P1_white[1])
            p2x = max(P2_black[0], P2_white[0])
            p2y = max(P2_black[1], P2_white[1])
            crop_top = image[p1y:p2y, p1x:p2x]
            if self.crop_analyzer(crop_top):
                # If most of the crop is black:
                p1_y = max(P1_black[1], P1_white[1])
            else:
                p1_y = min(P1_black[1], P1_white[1])

        # P4 - X
        if (abs(P4_black[0] - P4_white[0])) <= threshold:
            p4_x = min(P4_black[0], P4_white[0])
        # Analyze the right crop.
        else:
            p2x = min(P2_black[0], P2_white[0])
            p2y = min(P2_black[1], P2_white[1])
            p4x = max(P4_black[0], P4_white[0])
            p4y = max(P4_black[1], P4_white[1])
            crop_right = image[p2y:p4y, p2x:p4x]
            if self.crop_analyzer(crop_right):
                # If most of the crop is black:
                p4_x = min(P4_black[0], P4_white[0])
            else:
                p4_x = max(P4_black[0], P4_white[0])
        # P4 - Y
        if (abs(P4_black[1] - P4_white[1])) <= threshold:
            p4_y = min(P4_black[1], P4_white[1])
        # Analyze the bottom crop.
        else:
            p3x = min(P3_black[0], P3_white[0])
            p3y = min(P3_black[1], P3_white[1])
            p4x = max(P4_black[0], P4_white[0])
            p4y = max(P4_black[1], P4_white[1])
            crop_bottom = image[p3y:p4y, p3x:p4x]
            if self.crop_analyzer(crop_bottom):
                # If most of the crop is black:
                p4_y = min(P4_black[1], P4_white[1])
            else:
                p4_y = max(P4_black[1], P4_white[1])

        box = np.array([p1_x, p1_y, p4_x, p4_y], dtype=np.int32)

        return box

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
        resized = image.copy()

        if not self.white_border_method:
            if self.reading_raw_images:
                box = self.dark_approach(original_image)
                if (
                    box[2] - box[0] > 0.865 * original_image.shape[1]
                    or box[3] - box[1] > 0.865 * original_image.shape[0]
                ):
                    bright_box = self.bright_approach(image)
                    box = self.first_box_verification(box, bright_box)
            else:
                box = self.dark_approach(image)
                if (
                    box[2] - box[0] > 0.865 * original_image.shape[1]
                    or box[3] - box[1] > 0.865 * original_image.shape[0]
                ):
                    bright_box = self.bright_approach(image)
                    box = self.first_box_verification(box, bright_box)

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
                5,
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
                5,
            )

            cv2.namedWindow("Both Methods", 0)
            cv2.imshow("Both Methods", image)
            k = cv2.waitKey(0)
            if k == ord("q"):
                exit()

        if not self.white_border_method:
            # Compare the detected bounding boxes of both methods and returns the appropriate box.
            box = self.box_verification(box, white_box, resized)
        else:
            box = white_box

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
        increase_crop_y = int(cropped.shape[0] * self.overcrop)

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
                self.reading_raw_images = True
                with rawpy.imread(image_path) as raw:
                    image = raw.postprocess()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Read compressed images with OpenCV
            else:
                self.reading_raw_images = False
                image = cv2.imread(image_path, 1)

            image_name = image_path.split(sp)[-1].split(".")[0]
            image_name = image_name.replace(" ", "_")
            image_name = image_name.replace("'", "_")
            image_name = image_name.replace(",", "_")
            new_path = os.path.join(self.output_dir, image_name + ".jpg")

            # Remove the black borders
            cropped = self.remove_borders(image)

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
        default=1.1,
        help="Increase the crop based on percentage value (the overcrop between 0 and 100), defaults to 5",
    )
    parser.add_argument(
        "-w",
        "--white_border",
        action="store_true",
        help="Change the method to crop images with white borders.",
    )

    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    overcrop = args.overcrop / 100
    white_border = args.white_border

    os.makedirs(output, exist_ok=True)

    croper = ImageCropper(directory, output, overcrop, white_border)
    croper.crop()
