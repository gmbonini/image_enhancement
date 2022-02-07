import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import imutils
from torch_model.model import MobileNet
from argparse import ArgumentParser
from imutils.paths import list_images
from tqdm import tqdm


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
        self.out_dir = out_dir
        self.image_list = list(list_images(data_dir))

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
            cv2.imwrite(new_path, res)


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
