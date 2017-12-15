from __future__ import print_function
import numpy as np
import scipy.io
from PIL import Image, ImageOps, ImageFile
import scipy.misc
import tensorflow as tf

IMAGE_SIZE = 256


def get_resized_image(img_path, height, width):
    image = Image.open(img_path)
    # it's because PIL is column major so you have to change place of width and height
    # this is stupid, i know
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)


def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0]  # the image
    image = np.clip(image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, image)


def generate_noise_image(content_image, height, width, batch_size, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20,
                                    (batch_size, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)






