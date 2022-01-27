# README

This repository contains Python scripts to to enhance image quality and crop out black borders from images.

The scripts:

-   image_cropper.py: removes black borders from pictures by detecting them and cropping only the image itself.

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

*   --image_dir: path pointing to the directory containg all image files (.jpeg, .jpg, .png or .cr3 files are supported);
*   --out_dir: path pointing to the desired output directory, where the fixed videos or enhanced images will be saved.

The image_stitcher script has only two arguments: video_dir and out_dir. You can run it like this:

Likewise, the image_cropper.py script also has only two arguments: image_dir and out_dir. You can run it like this:

```shell
python image_cropper.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH
```
