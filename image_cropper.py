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

    def bright_approach(self, image, th=10, draw_box=False):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly bright
        """

        # Copy the original image
        original_image = image.copy()

        # Set a 60% of the overcrop percentage
        self.overcrop_black_box = self.default_overcrop * 0.6

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize the image with threshold
        _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

        # Dilate the threshold result - increase the white region in the image or the size of the foreground object
        dilate = cv2.dilate(
            thresh,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        # Erodes the dilated image - the thickness or size of the foreground object decreases
        erosion = cv2.erode(
            dilate, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
        )

        # Remove noise of the eroded image
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

        if box is not None and draw_box:
            cv2.rectangle(
                original_image,
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
            cv2.namedWindow("Bright", 0)
            cv2.imshow("Bright", original_image)

        return box

    def dark_approach(self, image, draw_box=False):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly dark
        """

        # Set a 100% of the overcrop percentage
        self.overcrop_black_box = self.default_overcrop

        # Convert the image to the HSV colorspace
        res = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split the image channels
        hue, sat, val = cv2.split(res)

        # Erodes the HUE channel of the image - the thickness or size of the foreground object decreases
        erosion = cv2.erode(
            hue, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
        )

        # Remove noise of the eroded image
        open = cv2.morphologyEx(
            erosion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 6))
        )

        # Dilate the opening image - increase the white region in the image or the size of the foreground object
        dilate = cv2.dilate(
            open,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        if self.reading_raw_images:
            # Find the contours of the *dilatation* in case of RAW images
            contours = cv2.findContours(
                dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = imutils.grab_contours(contours)
        else:
            # Find the contours of the *hue channel* in case of JPEG images
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
            # In case of RAW images, apply a resize
            resized_image = cv2.resize(image, (1280, 720))
            resized_image_copy = resized_image.copy()

            normalized_box = []
            normalized_box.append(box[0] / image.shape[1])
            normalized_box.append(box[1] / image.shape[0])
            normalized_box.append(box[2] / image.shape[1])
            normalized_box.append(box[3] / image.shape[0])

            # Normalize the bounding box coordinates
            normalized_box[0] = int(normalized_box[0] * resized_image.shape[1])
            normalized_box[1] = int(normalized_box[1] * resized_image.shape[0])
            normalized_box[2] = int(normalized_box[2] * resized_image.shape[1])
            normalized_box[3] = int(normalized_box[3] * resized_image.shape[0])

            if box is not None and draw_box:
                cv2.rectangle(
                    resized_image_copy,
                    (
                        int(normalized_box[0]),
                        int(normalized_box[1]),
                    ),
                    (
                        int(normalized_box[2]),
                        int(normalized_box[3]),
                    ),
                    (0, 0, 255),
                    5,
                )

                cv2.namedWindow("Dark", 0)
                cv2.imshow("Dark", resized_image_copy)
            return normalized_box
        else:
            return box

    def calculate_pixel_percentage(self, image, p=0.05):

        # Calculate histogram
        s = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Calculate percentage of pixels where the first image channel >= p
        s_perc = np.sum(s[int(p * 255) : -1]) / np.prod(image.shape[0:2])

        return s_perc

    def crop_analyzer(self, crop):

        if not self.reading_raw_images:
            # Convert image to HSV color space
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(image)

            s_thr = 0.5
            s_perc = self.calculate_pixel_percentage(sat)

            # Percentage threshold; above: valid image, below: black image.
            if s_perc > s_thr:
                return False
            else:
                s_thr = 0.0015
                s_perc = self.calculate_pixel_percentage(val)
                # Percentage threshold; above: valid image, below: black image.
                if s_perc > s_thr:
                    return False
                else:
                    s_thr = 0.3
                    s_perc = self.calculate_pixel_percentage(val - sat)
                    if s_perc < s_thr:
                        return False
                    else:
                        return True
        else:
            # Convert image to HSV color space
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Split the image channels
            hue, sat, val = cv2.split(image)

            # Erodes the HUE channel of the image
            erosion = cv2.erode(
                hue, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
            )

            # Remove noise of the eroded image
            open = cv2.morphologyEx(
                erosion,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            )

            s_thr = 0.025
            s_perc = self.calculate_pixel_percentage(open)

            # Percentage threshold; above: valid image, below: black image.
            if s_perc > s_thr:
                return False
            else:
                s_thr = 0.4
                s_perc = self.calculate_pixel_percentage(val)

                # Percentage threshold; above: valid image, below: black image.
                if s_perc > s_thr:
                    return False
                else:
                    s_thr = 0.75
                    s_perc = self.calculate_pixel_percentage(val - sat)

                    # Percentage threshold; below: valid image, above: black image.
                    if s_perc < s_thr:
                        return False
                    else:
                        return True

    def box_verification(self, black_box, white_box, image, threshold=30):

        """
        This functions compare the size of the bounding boxes and returns the
        the appropriate bounding box to use for crop.
        """

        if black_box is None:
            return white_box

        if white_box is None:
            return black_box

        # Get the bounding box vertex - black borders method
        P1_black = [black_box[0], black_box[1]] # Top left
        P2_black = [black_box[2], black_box[1]] # Top right
        P3_black = [black_box[0], black_box[3]] # Bottom left
        P4_black = [black_box[2], black_box[3]] # Bottom right

        # Get the bounding box vertex - white borders method
        P1_white = [white_box[0], white_box[1]] # Top left
        P2_white = [white_box[2], white_box[1]] # Top right
        P3_white = [white_box[0], white_box[3]] # Bottom left
        P4_white = [white_box[2], white_box[3]] # Bottom right

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

        # Copy the original image
        original_image = image.copy()

        # Apply a resize on the image, to improove the border detection
        image = cv2.resize(image, (1280, 720))

        # Copy the resized image
        resized = image.copy()

        # If are not using the white borders detection method
        if not self.white_border_method:

            # If are reading RAW images
            if self.reading_raw_images:

                # Pass the original image to dark approach
                dark_box = self.dark_approach(original_image)

                # Pass the resized image to bright approach
                bright_box = self.bright_approach(image)

                # If the dark approach bounding box occupies more than 86% of the original image size,
                # set the bright approach bounding box coordinates
                if (
                    dark_box[2] - dark_box[0] > 0.865 * original_image.shape[1]
                    or dark_box[3] - dark_box[1] > 0.865 * original_image.shape[0]
                ):
                    box = bright_box
                # If the bright approach bounding box occupies more than 86% of the original image size,
                # set the dark approach bounding box coordinates
                elif (
                    bright_box[2] - bright_box[0] > 0.865 * image.shape[1]
                    or bright_box[3] - bright_box[1] > 0.865 * image.shape[0]
                ):
                    box = dark_box
                # Choose the most appropriate bounding box coorditates, between the dark and bright approach
                else:
                    box = self.box_verification(dark_box, bright_box, image)
            else:
                # Pass the resized image to dark approach
                dark_box = self.dark_approach(image)

                # Pass the resized image to bright approach
                bright_box = self.bright_approach(image)

                # If the dark approach bounding box occupies more than 86% of the resized image size,
                # set the bright approach bounding box coordinates
                if (
                    dark_box[2] - dark_box[0] > 0.865 * image.shape[1]
                    or dark_box[3] - dark_box[1] > 0.865 * image.shape[0]
                ):
                    box = bright_box
                # If the bright approach bounding box occupies more than 86% of the resized image size,
                # set the dark approach bounding box coordinates
                elif (
                    bright_box[2] - bright_box[0] > 0.865 * image.shape[1]
                    or bright_box[3] - bright_box[1] > 0.865 * image.shape[0]
                ):
                    box = dark_box
                # Choose the most appropriate bounding box coorditates, between the dark and bright approach
                else:
                    box = self.box_verification(dark_box, bright_box, image)

        # Detect the borders with the white border detection method
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
            # k = cv2.waitKey(0)
            # if k == ord("q"):
            #     exit()

        # If are not using the white borders detection method
        if not self.white_border_method:

            # Compare the detected bounding boxes of both methods and returns the appropriate box.
            box = self.box_verification(box, white_box, resized)
        else:
            # Set the white border bounding box coordinates
            box = white_box

        # Normalize the bounding boxes coordinates, because the cropped image will be the original image,
        # not the resized image
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

        # Increase the crop, according to the overcrop percentage
        increase_crop_x = int(cropped.shape[1] * self.overcrop)
        increase_crop_y = int(cropped.shape[0] * self.overcrop)

        cropped = cropped[
            increase_crop_y : cropped.shape[0] - increase_crop_y,
            increase_crop_x : cropped.shape[1] - increase_crop_x,
        ]

        return cropped

    def white_borders(self, image):

        # Detect the edges of the image
        img = cv2.Canny(image, 80, 150)

        # Dilate the detected edges - increase the white region in the image or the size of the foreground object
        dilate = cv2.dilate(
            img,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        # Find the contours of the dilated image
        cnts = cv2.findContours(
            image=dilate, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )

        # Grab the image contours
        cnts = imutils.grab_contours(cnts)

        # Sort the contours
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Get the largest contour found
        cnt = cnts[0]

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h])

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

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

            # Remove spaces, "'" and "," from the filename
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
