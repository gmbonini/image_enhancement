# README

This repository contains Python scripts to to enhance image quality, crop out black borders from images and fix vertical syncrhonization issues on videos.

The scripts:

-   image_stitcher.py: fixes vertical synchronization issues on videos by detecting the desynchronized parts and stacking them together.
-   image_cropper.py: removes black borders from pictures by detecting them and cropping only the image itself.

### Installation

Create a Anaconda virtual environment with Python 3.8+:

```
conda create --name enhance python=3.8
conda activate enhance
```

Install the required Python modules

```
pip install -r requirements.txt
```

### Run the scripts

    Each script can be run individually from the command line and have the following possible arguments:

    -   --video_dir: path pointing to the directory containg all video files (.mp4, .avi or .ts files are supported);
    -   --image_dir: path pointing to the directory containg all image files (.jpeg, .jpg, .png or .cr3 files are supported);
    -   --out_dir: path pointing to the desired output directory, where the fixed videos or enhanced images will be saved.

    The image_stitcher script has only two arguments: video_dir and out_dir. You can run it like this:

    ```
    python image_stitcher.py --video_dir VIDEO_DIR_PATH --out_dir OUT_DIR_PATH
    ```

    Likewise, the image_cropper.py script also has only two arguments: image_dir and out_dir. You can run it like this:

    ```
    python image_cropper.py --image_dir IMAGE_DIR_PATH --out_dir OUT_DIR_PATH
    ```
