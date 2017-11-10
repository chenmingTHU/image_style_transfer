import tensorflow as tf
import numpy as np
import preprocessing
import os.path
import glob
import os
import matplotlib.pyplot as plt
import LossFunctions


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def transfer_images_to_tfrecords(train_path, image_path, file_num):
    extensions = ['jpg', 'png']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_path, '*.' + extension)
        file_list.extend(glob.glob(file_glob))

    num_per_num = int(len(file_list) / file_num)
    for i in range(file_num):
        filename = (train_path + 'data.tfrecords-%.5d-of-%.5d' % (i, file_num))
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(num_per_num):
            index = i * num_per_num + j
            image = preprocessing.get_resized_image(file_list[index], preprocessing.IMAGE_SIZE, preprocessing.IMAGE_SIZE)
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()


def load_tfrecords_and_get_batch(train_path):
    # extract the data from tfrecords, and generate a batch

    file_list = tf.train.match_filenames_once(train_path + 'data.tfrecords-*')
    filename_queue = tf.train.string_input_producer(file_list, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'image_raw': tf.FixedLenFeature([], tf.string)})
    image = features['image_raw']
    decoded_image = tf.decode_raw(image, tf.int8)
    decoded_image.set_shape([preprocessing.IMAGE_SIZE, preprocessing.IMAGE_SIZE, 3])

    min_after_dequeue = 10000
    batch_size = 4
    capacity = min_after_dequeue + 3 * batch_size
    image_batch = tf.train.shuffle_batch(decoded_image, batch_size=batch_size, capacity=capacity,
                                         min_after_dequeue=min_after_dequeue)
    return image_batch




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
    images = tf.image.resize_image_with_crop_or_pad(images, height, width)
    # images = tf.image.resize_images(images, size=[height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_batch = tf.train.batch([images], batch_size=batch_size, num_threads=5, capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch



# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i<2:
#
#             image = sess.run(image_batch)
#
#             for j in range(2):
#                 plt.imshow(image[j])
#                 plt.show()
#
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)

