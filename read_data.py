import tensorflow as tf
import numpy as np
import preprocessing
import os.path
import os


def get_batch(image_path, batch_size, capacity):

    def get_file(image_path):
        file_list = []
        for file in os.listdir(image_path):
            file_list.append(image_path + file)
        temp = np.array(file_list)
        np.random.shuffle(temp)
        image_list = list(temp)
        return image_list


    height = preprocessing.IMAGE_SIZE
    width = preprocessing.IMAGE_SIZE

    image_list = get_file(image_path)
    image_list = tf.cast(image_list, tf.string)
    input_queue = tf.train.slice_input_producer([image_list])
    image_contents = tf.read_file(input_queue[0])

    images = tf.image.decode_jpeg(image_contents, channels=3)
    # images = tf.image.resize_image_with_crop_or_pad(images, height, width)
    images = tf.image.resize_images(images, size=[height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_batch = tf.train.batch([images], batch_size=batch_size, num_threads=5, capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch


