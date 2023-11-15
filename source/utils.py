import os
import numpy as np


def remove_old_files(output_path):
    for f in os.listdir(output_path):
        os.remove(os.path.join(output_path, f))


def normalization_automatic(image):
    return normalization(image, np.min(image), np.max(image))


def normalization(image, image_min, image_max):
    new_image = image.copy().astype(float)
    new_image[image > image_max] = image_max
    new_image[image < image_min] = image_min
    new_image = (new_image - image_min) / (image_max - image_min)
    return new_image
