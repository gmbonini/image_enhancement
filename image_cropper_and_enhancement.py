import os
import cv2
import numpy as np
import imutils
import rawpy

from argparse import ArgumentParser
from glob import glob
from imutils.paths import list_images
from tqdm import tqdm
from enhance_image_skimage import Enhancer
from image_cropper import ImageCropper

sp = os.path.sep


class EnhanceCrop:
    def __init__(
        self, directory: str, output_dir: str, overcrop: float, white_border
    ) -> None:

        self.directory = directory
        self.output_dir = output_dir
        self.default_overcrop = overcrop
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

    def crop(self):

        parameters = {}
        parameters["local_contrast"] = 1.0     # no increase in details
        parameters["mid_tones"] = 0.2          # middle of range
        parameters["tonal_width"] = 0.3        # middle of range
        parameters["areas_dark"] = 0.2         # no change in dark areas
        parameters["areas_bright"] = 0.0       # no change in bright areas
        parameters["saturation_degree"] = 0.7  # no change in color saturation
        parameters["brightness"] = 0.2         # increase overall brightness
        parameters["preserve_tones"] = True
        parameters["color_correction"] = True

        enhancer = Enhancer(parameters)
        croper = ImageCropper(self.directory, self.output_dir, self.default_overcrop, self.white_border_method)

        for image_path in tqdm(self.images_list):

            # Convert raw images (CR3 files) to numpy arrays
            if image_path.lower().endswith(".cr3"):
                croper.reading_raw_images = True
                with rawpy.imread(image_path) as raw:
                    image = raw.postprocess()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Read compressed images with OpenCV
            else:
                croper.reading_raw_images = False
                image = cv2.imread(image_path, 1)

            image_name = image_path.split(sp)[-1].split(".")[0]
            image_name = image_name.replace(" ", "_")
            image_name = image_name.replace("'", "_")
            image_name = image_name.replace(",", "_")
            new_path = os.path.join(self.output_dir, image_name + ".jpg")

            # Remove the black borders
            cropped = croper.remove_borders(image)

            enhanced = enhancer.enhance(cropped)
            enhanced = np.ascontiguousarray(enhanced) * 255.0
            enhanced = enhanced.astype(np.uint8)

            # Save the new image
            cv2.imwrite(new_path, enhanced)

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

    croper = EnhanceCrop(directory, output, overcrop, white_border)
    croper.crop()
