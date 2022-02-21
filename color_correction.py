import os
import cv2
import numpy as np

from fileinput import filename
from argparse import ArgumentParser
from enlighten_inference import EnlightenOnnxModel
from tqdm import tqdm
from imutils.paths import list_images


class ImageColorCorrection:
    def __init__(self, directory: str, output_dir: str, gan_method, image_proc_method, white_balance) -> None:
        self.directory = directory
        self.output_dir = output_dir
        self.model = EnlightenOnnxModel()
        self.alpha = 1.0
        self.beta = 15
        self.gamma = 0.8
        self.gan_method = gan_method
        self.image_proc_method = image_proc_method
        self.apply_white_balance = white_balance

        if self.image_proc_method == False and  self.image_proc_method == False:
            self.image_proc_method = True

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
        
        images = list(list_images(self.directory))
        
        for image_path in tqdm(images):

            img = cv2.imread(image_path)

            if self.image_proc_method:
                brightness_corrected = self.brightness_correction(img, alpha=self.alpha, beta=self.beta)
                gamma_corrected = self.gamma_correction(brightness_corrected, self.gamma)
                saturation_corrected = self.saturation_correction(gamma_corrected)
                result = saturation_corrected

                # Windows path
                if "\\" in image_path:
                    new_path = os.path.join(
                        self.output_dir, image_path.split("\\")[-1].split(".")[0] + "_image_processing.jpg"
                )

                # Linux Path
                if "/" in image_path:
                    new_path = os.path.join(
                        self.output_dir, image_path.split("/")[-1].split(".")[0] + "_image_processing.jpg"
                )

                cv2.imwrite(new_path, result)              # HSV image

                if  self.apply_white_balance:
                    white_balanced = self.white_balance_correction(saturation_corrected)
                    cv2.imwrite(new_path, white_balanced)  # HSV with white balance

            if self.gan_method: 
                model_res = self.model.predict(img.copy())
                model_res = np.ascontiguousarray(model_res, dtype=np.uint8)

                # Windows path
                if "\\" in image_path:
                    new_path = os.path.join(
                        self.output_dir, image_path.split("\\")[-1].split(".")[0] + "_gan.jpg"
                )

                # Linux Path
                if "/" in image_path:
                    new_path = os.path.join(
                        self.output_dir, image_path.split("/")[-1].split(".")[0] + "_gan.jpg"
                )

                cv2.imwrite(new_path, model_res)           # GAN

                if  self.apply_white_balance:
                    model_res_white = self.white_balance_correction(model_res)
                    model_res_white = np.ascontiguousarray(model_res_white, dtype=np.uint8)
                    cv2.imwrite(new_path, model_res_white) # GAN with white balance

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
        "-g",
        "--gan",
        action='store_true',
        help="Uses the Generative Adversarial Network to enhance the image.",
    )
    parser.add_argument(
        "-p",
        "--image_proc",
        action='store_true',
        help="Uses image processing to enhance the image.",
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
    gan = args.gan
    image_proc = args.image_proc
    white_balance = args.white_balance
    os.makedirs(output, exist_ok=True)

    color_correction = ImageColorCorrection(directory, output, gan, image_proc, white_balance)
    color_correction.image_color_correction()

