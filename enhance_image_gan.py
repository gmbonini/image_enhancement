import os
import cv2
import numpy as np
import rawpy

from fileinput import filename
from argparse import ArgumentParser
from enlighten_inference import EnlightenOnnxModel
from tqdm import tqdm
from imutils.paths import list_images
from glob import glob

sp = os.path.sep


class ImageColorCorrectionGAN:
    def __init__(self, directory: str, output_dir: str, white_balance, resize: float) -> None:
        self.directory = directory
        self.output_dir = output_dir
        self.alpha = 1.0
        self.beta = 15
        self.gamma = 0.8
        self.resize_percentage = resize
        self.apply_white_balance = white_balance

        self.model = EnlightenOnnxModel()

    def image_color_correction(self):
        
        # List all images that are saved in a compressed format (JPEG, PNG, etc)
        compressed_images_list = list(list_images(self.directory))

        # List all images saved in a raw format with capitalized characters (CR3)
        raw_images_list = list(glob(os.path.join(self.directory, "*.CR3")))

        # List all images saved in a raw format with lowercase characters (cr3)
        raw_images_list_lower = list(glob(os.path.join(self.directory, "*.cr3")))

        # Sum all the image lists
        images = (
            set(compressed_images_list + raw_images_list + raw_images_list_lower)
        )
        
        for image_path in tqdm(images):
            # Convert raw images (CR3 files) to numpy arrays
            if image_path.lower().endswith(".cr3"):
                with rawpy.imread(image_path) as raw:
                    img = raw.postprocess()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Read compressed images with OpenCV
            else:
                img = cv2.imread(image_path, 1)

            dim = (int(img.shape[1]*self.resize_percentage), int(img.shape[0]*self.resize_percentage))
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            model_res = self.model.predict(img)
            model_res = np.ascontiguousarray(model_res, dtype=np.uint8)
            
            image_name = image_path.split(sp)[-1].split(".")[0]
            image_name = image_name.replace(" ","_")
            image_name = image_name.replace("'","_")
            image_name = image_name.replace(",","_")
            
            new_path = os.path.join(self.output_dir, image_name + ".jpg")

            cv2.imwrite(new_path, model_res)           # GAN

            if  self.apply_white_balance:
                model_res_white = self.white_balance_correction(model_res)
                model_res_white = np.ascontiguousarray(model_res_white, dtype=np.uint8)
                cv2.imwrite(new_path, model_res_white) # GAN with white balance
        
        print(f"Enhanced pictures with GAN saved to {self.output_dir}")

        return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        help="Path to the directory containing the image files.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path to the output directory, where the images will be saved.",
    )
    parser.add_argument(
        "-w",
        "--white_balance",
        action='store_true',
        help="Apply a white balance adjustment to the image.",
    )
    parser.add_argument(
        "-r",
        "--resize",
        type=float,
        default=0.8,
        help="Apply a risize to reduce the image size. The default value (0.8) generate an image with 80 percent of the original size.",
    )
    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    white_balance = args.white_balance
    resize = args.resize
    os.makedirs(output, exist_ok=True)

    color_correction = ImageColorCorrectionGAN(directory, output, white_balance, resize)
    color_correction.image_color_correction()

