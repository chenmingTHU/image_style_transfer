import tensorflow as tf


def conv2d(x, input_channel, output_channel, kernel_size, stride, mode='REFLECT'):
    with tf.variable_scope('conv'):

        shape = [kernel_size, kernel_size, input_channel, output_channel]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel_size / 2), int(kernel_size / 2)], [int(kernel_size / 2), int(kernel_size / 2)], [0, 0]],
                          mode=mode)
        conv = tf.nn.conv2d(x_padded, weight, strides=[1, stride, stride, 1], padding='VALID', name='conv')
        bias = tf.Variable(tf.zeros([output_channel]), name='bias')
        return tf.nn.bias_add(conv, bias)


def nn_upsample(input_tensor):

    height = tf.shape(input_tensor)[1] * 2
    width = tf.shape(input_tensor)[2] * 2

    upsampled_tensor = tf.image.resize_images(input_tensor, [height, width],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return upsampled_tensor

def relu(input_tensor):
    with tf.variable_scope('relu'):
        relu_tmp = tf.nn.relu(input_tensor)
        return relu_tmp


def decoder(input_tensor):


    with tf.variable_scope('conv4_1'):
        conv4_1 = conv2d(input_tensor, input_channel=512, output_channel=256, kernel_size=3, stride=1)
        relu4_1 = relu(conv4_1)
    with tf.variable_scope('upsample1'):
        ups_1 = nn_upsample(relu4_1)
    with tf.variable_scope('conv3_4'):
        conv3_4 = conv2d(ups_1, input_channel=256, output_channel=256, kernel_size=3, stride=1)
        relu3_4 = relu(conv3_4)
    with tf.variable_scope('conv3_3'):
        conv3_3 = conv2d(relu3_4, input_channel=256, output_channel=256, kernel_size=3, stride=1)
        relu3_3 = relu(conv3_3)
    with tf.variable_scope('conv3_2'):
        conv3_2 = conv2d(relu3_3, input_channel=256, output_channel=256, kernel_size=3, stride=1)
        relu3_2 = relu(conv3_2)
    with tf.variable_scope('conv3_1'):
        conv3_1 = conv2d(relu3_2, input_channel=256, output_channel=128, kernel_size=3, stride=1)
        relu3_1 = relu(conv3_1)
    with tf.variable_scope('upsample2'):
        ups_2 = nn_upsample(relu3_1)
    with tf.variable_scope('conv2_2'):
        conv2_2 = conv2d(ups_2, input_channel=128, output_channel=128, kernel_size=3, stride=1)
        relu2_2 = relu(conv2_2)
    with tf.variable_scope('conv2_1'):
        conv2_1 = conv2d(relu2_2, input_channel=128, output_channel=64, kernel_size=3, stride=1)
        relu2_1 = relu(conv2_1)
    with tf.variable_scope('upsample3'):
        ups_3 = nn_upsample(relu2_1)
    with tf.variable_scope('conv1_2'):
        conv1_2 = conv2d(ups_3, input_channel=64, output_channel=64, kernel_size=3, stride=1)
        relu1_2 = relu(conv1_2)
    with tf.variable_scope('conv1_1'):
        conv1_1 = conv2d(relu1_2, input_channel=64, output_channel=3, kernel_size=3, stride=1)

    return conv1_1
