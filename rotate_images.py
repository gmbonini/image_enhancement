from http.client import OK
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import imutils
import pdb
from torch_model.model import MobileNet
from argparse import ArgumentParser
from imutils.paths import list_images
from tqdm import tqdm
import rawpy


class RotationModel(object):
    def __init__(self, data_dir, out_dir, model_path):

        self.angles = self.angles = [0, 90, 180, 270]
        self.idx_to_angle = {idx: angle for idx, angle in enumerate(self.angles)}
        self.angle_to_idx = {angle: idx for idx, angle in enumerate(self.angles)}
        self.input_shape = [224, 224, 3]
        state_dict = dict(torch.load(model_path)["model_state_dict"])

        self.model = MobileNet(self.input_shape, num_classes=4)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.cuda()

        self.image_list = list(list_images(data_dir))
        self.out_dir = out_dir
        self.overcrop = 0.95

    def preprocess(self, image: np.ndarray):

        pre = cv2.resize(image, tuple(self.input_shape[:2]))
        pre = pre / 255.0
        pre = pre.astype(np.float32)
        pre = np.expand_dims(pre, axis=0)
        pre = torch.tensor(pre)
        pre = pre.permute((0, 3, 1, 2))
        return pre

    def inference(self, image):

        pre = self.preprocess(image).cuda()
        out = self.model(pre)

        pred = out.detach().cpu().numpy()
        pred = F.log_softmax(out, dim=1)
        pred = pred.argmax(dim=1)

        angle = self.idx_to_angle[pred.item()]

        rot = imutils.rotate_bound(image, -angle)

        return rot

    def run(self):
        for image_path in tqdm(self.image_list):
            img = cv2.imread(image_path, 1)
            res = self.inference(img)
            
            # Save the new image
            new_path = os.path.join(
                self.out_dir, image_path.split("/")[-1].split(".")[0] + ".jpg"
            )
            warp = self.find_borders(res)
            
            cv2.imwrite(new_path, warp)


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

        ((x, y), (w, h), angle) = cv2.minAreaRect(contour)

        box_points = np.int0(
            cv2.boxPoints(
                ((x, y), (int(w * self.overcrop), int(h * self.overcrop)), angle)
            )
        )

        """
        OpenCV's boxPoints function doesn't return the points in the same order all the time
        So we need re-order them to match our reference:
        
        box = [ 
            [x, y], # Bottom right
            [x, y], # Bottom left
            [x, y], # Top left
            [x, y]  # Top right
        ]
        """
        sort_by_x_indexes = np.argsort(box_points[:, 0])
        sorted_by_x = box_points[sort_by_x_indexes, :]
        left_points = sorted_by_x[:2, :]
        right_points = sorted_by_x[2:, :]

        left_sort_by_y_indexes = np.argsort(left_points[:, 1])
        top_left = left_points[left_sort_by_y_indexes[0]]
        bottom_left = left_points[left_sort_by_y_indexes[1]]

        right_sort_by_y_indexes = np.argsort(right_points[:, 1])
        top_right = right_points[right_sort_by_y_indexes[0]]
        bottom_right = right_points[right_sort_by_y_indexes[1]]
        
        box_points = [
            bottom_right,
            bottom_left,
            top_left,
            top_right
        ]
        box_points = np.array(box_points, dtype=np.int32)
        
        return box_points, angle

    def dark_approach(self, image):

        """
        This function is responsible for estimating the bounding box of the picture
        (without the black borders) if it's mostly dark
        """

        # Convert the image to the HSV color space
        res = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, _, _ = cv2.split(res)

        # Find the contours of the saturation channel
        contours = cv2.findContours(hue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        # Get the bounding boxes for the most significant contours
        boxes = []
        for contour in contours:
            ((x, y), (w, h), angle) = cv2.minAreaRect(contour)

            if w > image.shape[1] * 0.2 and h > image.shape[0] * 0.2:
                box_points = np.int0(
                    cv2.boxPoints(
                        (
                            (x, y),
                            (int(w * self.overcrop), int(h * self.overcrop)),
                            angle,
                        )
                    )
                )

                
                """
                OpenCV's boxPoints function doesn't return the points in the same order all the time
                So we need re-order them to match our reference:
                
                box = [ 
                    [x, y], # Bottom right
                    [x, y], # Bottom left
                    [x, y], # Top left
                    [x, y]  # Top right
                ]
                """

                sort_by_x_indexes = np.argsort(box_points[:, 0])
                sorted_by_x = box_points[sort_by_x_indexes, :]
                left_points = sorted_by_x[:2, :]
                right_points = sorted_by_x[2:, :]

                left_sort_by_y_indexes = np.argsort(left_points[:, 1])
                top_left = left_points[left_sort_by_y_indexes[0]]
                bottom_left = left_points[left_sort_by_y_indexes[1]]

                right_sort_by_y_indexes = np.argsort(right_points[:, 1])
                top_right = right_points[right_sort_by_y_indexes[0]]
                bottom_right = right_points[right_sort_by_y_indexes[1]]
                
                box_points = [
                    bottom_right,
                    bottom_left,
                    top_left,
                    top_right
                ]
                box_points = np.array(box_points, dtype=np.int32)
                boxes.append(box_points)
        boxes = np.array(boxes, dtype=np.int32)

        # Get the bounding box that encloses all the bounding boxes
        if len(boxes) == 1:
            box = boxes[0]
        elif len(boxes) == 0:
            box = None
        else:
            # First vertex [x, y]
            bottom_right_x = np.max(boxes[:, 0, 0])
            bottom_right_y = np.max(boxes[:, 0, 1])

            bottom_left_x = np.min(boxes[:, 1, 0])
            bottom_left_y = np.max(boxes[:, 1, 1])

            top_left_x = np.min(boxes[:, 2, 0])
            top_left_y = np.min(boxes[:, 2, 1])

            top_right_x = np.max(boxes[:, 3, 0])
            top_right_y = np.min(boxes[:, 3, 1])

            box = np.array(
                [
                    [bottom_right_x, bottom_right_y],
                    [bottom_left_x, bottom_left_y],
                    [top_left_x, top_left_y],
                    [top_right_x, top_right_y],
                ],
                dtype=np.int32,
            )

        return box, angle

    def find_borders(self, image: np.ndarray) -> np.ndarray:

        """
        This functions takes an image, binarize it to get what is an image
        and what is just a black border and get the image contours
        """

        # First, try the dark approach
        box, angle = self.dark_approach(image)

        # If the dark approach resulted in a bounding box that encloses almost
        # all the image, switch to the bright approach

        width = box[0, 0] - box[2, 0] # Bottom right X - Top left X
        height = box[0, 1] - box[2, 1] # Bottom right Y - Top left Y

        if box is None:
            box, angle = self.bright_approach(image)
        elif (
            width > 0.865 * image.shape[1]
            or height > 0.865 * image.shape[0]
        ):
            box, angle = self.bright_approach(image)

        print(f"BBox angle: {angle:.4f}")
        # cv2.drawContours(image, [box], 0, (36, 255, 12), 5)
        frame_points = np.array([[width, height], [0, height], [0, 0], [width, 0]])
        transform = cv2.getAffineTransform(box[:-1].astype(np.float32), frame_points[:-1].astype(np.float32))
        warp = cv2.warpAffine(image, transform, (width, height))

        return warp


if __name__ == "__main__":

    MODEL_PATH = "./rotation_model/epoch_32.pth"

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

    evaluator = RotationModel(args.image_dir, args.out_dir, MODEL_PATH)
    evaluator.run()
