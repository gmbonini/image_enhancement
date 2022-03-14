# README

This repository contains Python scripts to enhance image quality and crop out black borders from images.

The scripts:

-   image_cropper.py: Removes borders from pictures by detecting them and cropping only the image itself. This script have two methods for borders detection, the default method detect black borders and an the optional method detect white borders. 
-   enhance_image_opencv.py: Make adjustments in the image color using only image processing techniques. 
-   enhance_image_gan.py: Make adjustments in the image color using a Deep Learning model that is able to generate images, to perform a fully-automatic color correction and detail enhancement. 
-   enhance_image_skimage.py: Apply a complex image enhancement algorithm to a directory containing images.
-   image_cropper_and_enhancement.py: Remove borders from pictures by detecting them and cropping only the image itself and enhance the cropped image with the enhance_image_skimage method.

### Installation

Create a Anaconda virtual environment with Python 3.8+:

```shell
conda create --name enhance python=3.8
conda activate enhance
```

Install the required Python modules

```shell
pip install -r requirements.txt
```

### Run the scripts

Each script can be run individually from the command line and have the following possible arguments:

-   --image_dir: Path pointing to the directory containg all image files (.jpeg, .jpg, .png or .cr3 files are supported);
-   --out_dir: Path pointing to the desired output directory, where the cropped or enhanced images will be saved.

The `image_cropper` and `image_cropper_and_enhancement` has additional arguments:

-   --overcrop (**optional**): Increase the crop based on percentage value (the overcrop between 0 and 100). The default percentage is 1.1. If you want to use the default percentage, run without the argument. If you want to remove the overcrop, run with: `--overcrop 0`. 
-   --white_border (**optional**): Use only the method for `white borders` detection. 

The `enhance_image_opencv` and `enhance_image_gan` has an aditional argument:

-   --white_balance (**optional**): Apply a white balance adjustment to the image.

The `enhance_image_gan` has an aditional argument:

-   --resize: Apply a risize to reduce the image size. The default value (0.8) generate an image with 80 percent of the original size.

You can run the scripts like this:

```shell
python image_cropper.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH --overcrop 1.09 --white_border

python enhance_image_opencv.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH --white_balance

python enhance_image_gan.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH --white_balance  --resize 0.8

python enhance_image_skimage.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH

python image_cropper_and_enhancement.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH --overcrop 1.1 --white_border
```
