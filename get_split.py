"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import os  # for os based ops
import random  # for sampling
from pathlib import Path
from shutil import copy2  # to copy the images

from logger import logger

PATH_TO_DATASET = "./Datasets/CIFAR-10-images/"
NEW_PATH = "./Images_Test"
path_to_images = Path(PATH_TO_DATASET)

_all_images = [i for i in path_to_images.glob("*/*/*.jpg")]
print(len(_all_images))

sampled_images = random.sample(_all_images, k=60000)

os.makedirs(NEW_PATH, exist_ok=True)

for idx, i in enumerate(sampled_images):
    new_file_name = os.path.join(NEW_PATH, f"{idx}" + i.name)
    logger.info(f"Copying File {i} ====> {new_file_name}")

    copy2(i, new_file_name)

    if idx > 1000:
        break
