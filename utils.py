from pathlib import Path
from tqdm.auto import tqdm

import numpy as np


def get_video_files(video_dir: str) -> list:

    """
    Returns the list of video files (.mp4 or .ts) inside a directory.
    """

    video_files = (
        list(Path(video_dir).rglob("*.mp4"))
        + list(Path(video_dir).rglob("*.ts"))
        + list(Path(video_dir).rglob("*.avi"))
    )

    video_files = [str(video_file) for video_file in video_files]

    return video_files
