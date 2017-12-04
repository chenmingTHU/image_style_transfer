from __future__ import print_function
from PIL import Image, ImageOps, ImageFile
import scipy.misc
import scipy.io
import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np
import time
import sys


def adaIn(content_maps, style_maps):

    epsilon = 1e-9
    mean_content, var_content = tf.nn.moments(content_maps, [1, 2], keep_dims=True)
    mean_style, var_style = tf.nn.moments(style_maps, [1, 2], keep_dims=True)

    tmp = tf.div(tf.subtract(content_maps, mean_content), tf.sqrt(tf.add(var_content, epsilon)))
    res = tf.add(tf.multiply(tf.sqrt(tf.add(var_style, epsilon)), tmp), mean_style)

    return res


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



def conv2d_lossnet(input_tensor, vgg16_layers, layer_index, expected_layer_name):

    layer = vgg16_layers[0][layer_index][0][0]
    weight = layer[0][0][0]
    bias = layer[0][0][1]
    layer_name = layer[3]

    assert layer_name == expected_layer_name
    conv = tf.nn.conv2d(input_tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
    return conv + bias

def relu_lossnet(input_tensor, vgg16_layers, layer_index, expected_layer_name):

    layer = vgg16_layers[0][layer_index][0][0]
    layer_name = layer[1]

    assert layer_name == expected_layer_name
    relu = tf.nn.relu(input_tensor)
    return relu

def avgpool_lossnet(input_tensor):
    # we use average pooling instead of max pooling
    avgpool = tf.nn.avg_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return avgpool


def encoder(input_image):
    input_image = tf.pad(input_image, [[0, 0], [20, 20], [20, 20], [0, 0]], mode='REFLECT')

    vgg16_path = 'imagenet-vgg-verydeep-16.mat'
    vgg16 = scipy.io.loadmat(vgg16_path)
    vgg16_layers = vgg16['layers']

    model = {}

    model['conv1_1'] = conv2d_lossnet(input_image, vgg16_layers, 0, expected_layer_name='conv1_1')
    model['relu1_1'] = relu_lossnet(model['conv1_1'], vgg16_layers, 1, expected_layer_name='relu1_1')
    model['conv1_2'] = conv2d_lossnet(model['relu1_1'], vgg16_layers, 2, expected_layer_name='conv1_2')
    model['relu1_2'] = relu_lossnet(model['conv1_2'], vgg16_layers, 3, expected_layer_name='relu1_2')
    model['avgpool1'] = avgpool_lossnet(model['relu1_2'])

    model['conv2_1'] = conv2d_lossnet(model['avgpool1'], vgg16_layers, 5, expected_layer_name='conv2_1')
    model['relu2_1'] = relu_lossnet(model['conv2_1'], vgg16_layers, 6, expected_layer_name='relu2_1')
    model['conv2_2'] = conv2d_lossnet(model['relu2_1'], vgg16_layers, 7, expected_layer_name='conv2_2')
    model['relu2_2'] = relu_lossnet(model['conv2_2'], vgg16_layers, 8, expected_layer_name='relu2_2')
    model['avgpool2'] = avgpool_lossnet(model['relu2_2'])

    model['conv3_1'] = conv2d_lossnet(model['avgpool2'], vgg16_layers, 10, expected_layer_name='conv3_1')
    model['relu3_1'] = relu_lossnet(model['conv3_1'], vgg16_layers, 11, expected_layer_name='relu3_1')
    model['conv3_2'] = conv2d_lossnet(model['relu3_1'], vgg16_layers, 12, expected_layer_name='conv3_2')
    model['relu3_2'] = relu_lossnet(model['conv3_2'], vgg16_layers, 13, expected_layer_name='relu3_2')
    model['conv3_3'] = conv2d_lossnet(model['relu3_2'], vgg16_layers, 14, expected_layer_name='conv3_3')
    model['relu3_3'] = relu_lossnet(model['conv3_3'], vgg16_layers, 15, expected_layer_name='relu3_3')
    model['avgpool3'] = avgpool_lossnet(model['relu3_3'])

    model['conv4_1'] = conv2d_lossnet(model['avgpool3'], vgg16_layers, 17, expected_layer_name='conv4_1')
    model['relu4_1'] = relu_lossnet(model['conv4_1'], vgg16_layers, 18, expected_layer_name='relu4_1')
    model['conv4_2'] = conv2d_lossnet(model['relu4_1'], vgg16_layers, 19, expected_layer_name='conv4_2')
    model['relu4_2'] = relu_lossnet(model['conv4_2'], vgg16_layers, 20, expected_layer_name='relu4_2')
    model['conv4_3'] = conv2d_lossnet(model['relu4_2'], vgg16_layers, 21, expected_layer_name='conv4_3')
    model['relu4_3'] = relu_lossnet(model['conv4_3'], vgg16_layers, 22, expected_layer_name='relu4_3')
    model['avgpool4'] = avgpool_lossnet(model['relu4_3'])

    return model



def conv2d(x, input_channel, output_channel, kernel_size, stride, mode='REFLECT'):
    with tf.variable_scope('conv'):

        shape = [kernel_size, kernel_size, input_channel, output_channel]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1),dtype=tf.float32, name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel_size / 2), int(kernel_size / 2)], [int(kernel_size / 2), int(kernel_size / 2)], [0, 0]],
                          mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, stride, stride, 1], padding='VALID', name='conv')


# use NN upsampling to take place of pooling in encoder
def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):

    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def batch_norm(input_tensor, is_training, eps=0.0001, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='batch_norm'):
        shape = input_tensor.shape
        param_shape = shape[-1:]

        moving_mean = tf.get_variable('mean', param_shape, initializer=tf.zeros_initializer, trainable=False,
                                      dtype=tf.float32)
        moving_variance = tf.get_variable('variance', param_shape, initializer=tf.ones_initializer, trainable=False,
                                          dtype=tf.float32)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(input_tensor, [0, 1, 2], name='moments')
            with tf.control_dependencies([moving_averages.assign_moving_average(moving_mean, mean, decay),
                                          moving_averages.assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_variance))

        if affine:
            beta = tf.get_variable('beta', param_shape,
                                   initializer=tf.zeros_initializer, dtype=tf.float32)
            gamma = tf.get_variable('gamma', param_shape,
                                    initializer=tf.ones_initializer, dtype=tf.float32)
            output_tensor = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, eps)
        else:
            output_tensor = tf.nn.batch_normalization(input_tensor, mean, variance, None, None, eps)

        return output_tensor


def nn_upsample(input_tensor):
    shape = tf.shape(input_tensor)
    with tf.Session() as sess:
        shape = sess.run(shape)
    height = shape[1]
    width = shape[2]

    upsampled_tensor = tf.image.resize_images(input_tensor, [height*2, width*2],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return upsampled_tensor

def relu(input_tensor):
    with tf.variable_scope('relu'):
        relu_tmp = tf.nn.relu(input_tensor)
        # res = tf.where(tf.equal(relu_tmp, relu_tmp), relu_tmp, tf.zeros_like(relu_tmp))
        return relu_tmp


def sigmoid(input_tensor):
    with tf.variable_scope('sigmoid'):
        res = tf.nn.sigmoid(input_tensor)
        return res


def instance_norm(input_tensor):
    epsilon = 1e-9
    mean, var = tf.nn.moments(input_tensor, [1, 2], keep_dims=True)
    res = tf.div(tf.subtract(input_tensor, mean), tf.sqrt(tf.add(var, epsilon)))
    return res


def decoder(input_tensor, is_training=tf.constant(True, dtype=tf.bool), training=True):

    shape_x = input_tensor.shape
    height = shape_x[1] * 8
    width = shape_x[2] * 8

    with tf.variable_scope('conv_1'):
        conv_1 = conv2d(input_tensor, input_channel=512, output_channel=256, kernel_size=3, stride=1)
        # conv_1 = instance_norm(conv_1)
        conv_1 = batch_norm(conv_1, is_training)
        relu_1 = relu(conv_1)
    with tf.variable_scope('up_sample_1'):
        ups_1 = resize_conv2d(relu_1, input_filters=256, output_filters=256, kernel=3, strides=1, training=training)
        # ups_1 = nn_upsample(relu_1)
    with tf.variable_scope('conv_2'):
        conv_2 = conv2d(ups_1, input_channel=256, output_channel=256, kernel_size=3, stride=1)
        # conv_2 = instance_norm(conv_2)
        conv_2 = batch_norm(conv_2, is_training)
        relu_2 = relu(conv_2)
    with tf.variable_scope('conv_3'):
        conv_3 = conv2d(relu_2, input_channel=256, output_channel=256, kernel_size=3, stride=1)
        # conv_3 = instance_norm(conv_3)
        conv_3 = batch_norm(conv_3, is_training)
        relu_3 = relu(conv_3)
    with tf.variable_scope('conv_4'):
        conv_4 = conv2d(relu_3, input_channel=256, output_channel=128, kernel_size=3, stride=1)
        # conv_4 = instance_norm(conv_4)
        conv_4 = batch_norm(conv_4, is_training)
        relu_4 = relu(conv_4)
        relu_4 = relu_4 + conv2d(ups_1, input_channel=256, output_channel=128, kernel_size=1, stride=1)
    with tf.variable_scope('up_sample_2'):
        ups_2 = resize_conv2d(relu_4, input_filters=128, output_filters=128, kernel=3, strides=1, training=training)
        # ups_2 = nn_upsample(relu_4)
    with tf.variable_scope('conv_5'):
        conv_5 = conv2d(ups_2, input_channel=128, output_channel=128, kernel_size=3, stride=1)
        # conv_5 = instance_norm(conv_5)
        conv_5 = batch_norm(conv_5, is_training)
        relu_5 = relu(conv_5)
    with tf.variable_scope('conv_6'):
        conv_6 = conv2d(relu_5, input_channel=128, output_channel=64, kernel_size=3, stride=1)
        # conv_6 = instance_norm(conv_6)
        conv_6 = batch_norm(conv_6, is_training)
        relu_6 = relu(conv_6)
        relu_6 = relu_6 + conv2d(ups_2, input_channel=128, output_channel=64, kernel_size=1, stride=1)
    with tf.variable_scope('up_sample_3'):
        ups_3 = resize_conv2d(relu_6, input_filters=64, output_filters=64, kernel=3, strides=1, training=training)
        # ups_3 = nn_upsample(relu_6)
    with tf.variable_scope('conv_7'):
        conv_7 = conv2d(ups_3, input_channel=64, output_channel=64, kernel_size=3, stride=1)
        # conv_7 = instance_norm(conv_7)
        conv_7 = batch_norm(conv_7, is_training)
        relu_7 = relu(conv_7)
    with tf.variable_scope('conv_8'):
        conv_8 = conv2d(relu_7, input_channel=64, output_channel=3, kernel_size=3, stride=1)
        # conv_8 = instance_norm(conv_8)
        conv_8 = batch_norm(conv_8, is_training)
        relu_8 = tf.nn.tanh(conv_8)

    y = (relu_8 + 1) * 127.5

    shape_y = y.shape
    batch_size = shape_y[0]
    channels = shape_y[3]
    y = tf.slice(y, [0, 20, 20, 0], tf.stack([batch_size, height - 40, width - 40, channels]))

    return y





MEAN_PIXELS = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 1, 3))


def eval_arbitrary(content_path, style_path, output_path, height = 560, width = 800):

    content_image = get_resized_image(content_path, height, width)
    style_image = get_resized_image(style_path, height, width)

    content_map = encoder(content_image - MEAN_PIXELS)['relu4_1']
    style_map = encoder(style_image - MEAN_PIXELS)['relu4_1']

    fusion_map = adaIn(content_maps=content_map, style_maps=style_map)
    output_image = decoder(fusion_map, training=False, is_training=tf.constant(False, dtype=tf.bool))

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('arbitrary_model/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        res = sess.run(output_image)
        save_image(output_path, res)


content_path = 'content_test/001.jpg'
style_path = 'style_test/style15.jpg'
output_path = "result/result.jpg"
if __name__ == '__main__':
    #eval_arbitrary(content_path, style_path, output_path, height=560, width=800)
    eval_arbitrary(sys.argv[1], sys.argv[2], sys.argv[3], 560, 800)
    print("Finished.")
