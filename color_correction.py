import os
import cv2
import numpy as np

from fileinput import filename
from argparse import ArgumentParser
from enlighten_inference import EnlightenOnnxModel
from tqdm import tqdm
from imutils.paths import list_images


class ImageColorCorrection:
    def __init__(self, directory: str, output_dir: str) -> None:
        self.directory = directory
        self.output_dir = output_dir
        self.model = EnlightenOnnxModel()
        self.alpha = 1.0
        self.beta = 15
        self.gamma = 0.8

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
            image_name = image_path.split("/")[-1]

            brightness_corrected = self.brightness_correction(img, alpha=self.alpha, beta=self.beta)
            gamma_corrected = self.gamma_correction(brightness_corrected, self.gamma)
            saturation_corrected = self.saturation_correction(gamma_corrected)
            white_balanced = self.white_balance_correction(saturation_corrected)
            result = saturation_corrected

            directory_name = os.path.join(self.output_dir, image_name.split('.')[0])
            
            model_res = self.model.predict(img.copy())
            model_res_white = self.white_balance_correction(model_res)
            model_res = np.ascontiguousarray(model_res, dtype=np.uint8)
            model_res_white = np.ascontiguousarray(model_res_white, dtype=np.uint8)

            HSV_image_path = directory_name + "/" + "HSV_no_white_balance_" + image_name
            HSV_WB_image_path = directory_name + "/" + "HSV_with_white_balance_" + image_name
            GAN_image_path = directory_name + "/" + "GAN_no_white_balance_" + image_name
            GAN_WB_image_path = directory_name + "/" + "GAN_with_white_balance_" + image_name

            os.makedirs(directory_name, exist_ok=True)

            cv2.imwrite(HSV_image_path, result)             # HSV image
            cv2.imwrite(HSV_WB_image_path, white_balanced)  # HSV with white balance
            cv2.imwrite(GAN_image_path, model_res)          # GAN
            cv2.imwrite(GAN_WB_image_path, model_res_white) # GAN with white balance

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

    args = parser.parse_args()

    directory = args.image_dir
    output = args.out_dir
    os.makedirs(output, exist_ok=True)

    color_correction = ImageColorCorrection(directory, output)
    color_correction.image_color_correction()

