import os
import cv2
import numpy as np
import rawpy

from fileinput import filename
from argparse import ArgumentParser
from tqdm import tqdm
from imutils.paths import list_images
from glob import glob

sp = os.path.sep


class ImageColorCorrectionOpenCV:
    def __init__(self, directory: str, output_dir: str, white_balance) -> None:
        self.directory = directory
        self.output_dir = output_dir
        self.alpha = 1.0
        self.beta = 15
        self.gamma = 0.8
        self.apply_white_balance = white_balance

    def gamma_correction(self, img, gamma=0.8):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        result = cv2.LUT(img, lookUpTable)
        return result
        
    def brightness_correction(self, img, alpha=1.0, beta=15):
        result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return result

    def saturation_correction(self, img):
        # Converting image to HSV Color model 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

        # Splitting the HSV image to different channels
        h, s, v = cv2.split(hsv)

        # Applying CLAHE to S-channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6,6))
        c_hsv_s = clahe.apply(s)

        # Merge the CLAHE enhanced S-channel with the h and v channel
        hsv_img_s = cv2.merge((h, c_hsv_s, v))

        # Converting image from LAB Color model to BGR model
        result = cv2.cvtColor(hsv_img_s, cv2.COLOR_HSV2BGR)
        
        return result

    def white_balance_correction(self, img):
        b, g, r = cv2.split(img)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        # Find the gain of each channel
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg

        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        
        balance_processed = cv2.merge([b, g, r])

        return balance_processed

    def image_color_correction(self):
        
        # List all images that are saved in a compressed format (JPEG, PNG, etc)
        compressed_images_list = list(list_images(self.directory))

        # List all images saved in a raw format with capitalized characters (CR3)
        raw_images_list = list(glob(os.path.join(self.directory, "*.CR3")))

        # List all images saved in a raw format with lowercase characters (cr3)
        raw_images_list_lower = list(glob(os.path.join(self.directory, "*.cr3")))

        # Sum all the image lists
        images = (
            compressed_images_list + raw_images_list + raw_images_list_lower
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

            brightness_corrected = self.brightness_correction(img, alpha=self.alpha, beta=self.beta)
            gamma_corrected = self.gamma_correction(brightness_corrected, self.gamma)
            saturation_corrected = self.saturation_correction(gamma_corrected)
            result = saturation_corrected

            image_name = image_path.split(sp)[-1].split(".")[0]
            image_name = image_name.replace(" ","_")
            image_name = image_name.replace("'","_")
            image_name = image_name.replace(",","_")
            
            new_path = os.path.join(self.output_dir, image_name + ".jpg")

            cv2.imwrite(new_path, result)              # HSV image

            if  self.apply_white_balance:
                white_balanced = self.white_balance_correction(saturation_corrected)
                cv2.imwrite(new_path, white_balanced)  # HSV with white balance

        print(f"Enhanced pictures with OpenCV saved to {self.output_dir}")

        return


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
        help="Path to the output directory, where the images will be saved",
    )
    parser.add_argument(
        "-w",
        "--white_balance",
        action='store_true',
        help="Apply a white balance adjustment to the image.",
    )
    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    white_balance = args.white_balance
    os.makedirs(output, exist_ok=True)

    color_correction = ImageColorCorrectionOpenCV(directory, output, white_balance)
    color_correction.image_color_correction()

