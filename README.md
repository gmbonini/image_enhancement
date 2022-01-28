# README

This repository contains Python scripts to to enhance image quality and crop out black borders from images.

The scripts:

-   image_cropper.py: removes black borders from pictures by detecting them and cropping only the image itself.
-   color_correction.py: make adjustments in the image color with two approaches: 1 - Using only image processing techniques; 2 - Using a Deep Learning model that is able to generate images, to perform a fully-automatic color correction and detail enhancement.

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

-   --image_dir: path pointing to the directory containg all image files (.jpeg, .jpg, .png or .cr3 files are supported);
-   --out_dir: path pointing to the desired output directory, where the fixed videos or enhanced images will be saved.

You can run the scripts like this:

```shell
python image_cropper.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH
python color_correction.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH
```
