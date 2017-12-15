import tensorflow as tf
import numpy as np


WEIGHT_PATH = 'vgg19_normalised.npz'
ENCODER_LATERS = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                  'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1')

weights = np.load(WEIGHT_PATH)
idx = 0
weight_list = []

for layer in ENCODER_LATERS:
    kind = layer[:4]

    if kind == 'conv':
        kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
        bias = weights['arr_%d' % (idx + 1)]
        kernel = kernel.astype(np.float32)
        bias = bias.astype(np.float32)
        idx += 2

        with tf.variable_scope(layer):
            W = tf.Variable(kernel, trainable=False, name='kernel')
            b = tf.Variable(bias, trainable=False, name='bias')
        weight_list.append((W, b))


def conv2d(x, index):

    kernel, bias = weight_list[index]
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    res = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    res = tf.nn.bias_add(res, bias)

    return res


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def relu(x):
    return tf.nn.relu(x)


def encoder(input_tensor):

    model = {}

    model['conv1_1'] = conv2d(input_tensor, 0)
    model['relu1_1'] = relu(model['conv1_1'])

    model['conv1_2'] = conv2d(model['relu1_1'], 1)
    model['relu1_2'] = relu(model['conv1_2'])

    model['pool1'] = pool2d(model['relu1_2'])

    model['conv2_1'] = conv2d(model['pool1'], 2)
    model['relu2_1'] = relu(model['conv2_1'])

    model['conv2_2'] = conv2d(model['relu2_1'], 3)
    model['relu2_2'] = relu(model['conv2_2'])

    model['pool2'] = pool2d(model['relu2_2'])

    model['conv3_1'] = conv2d(model['pool2'], 4)
    model['relu3_1'] = relu(model['conv3_1'])

    model['conv3_2'] = conv2d(model['relu3_1'], 5)
    model['relu3_2'] = relu(model['conv3_2'])

    model['conv3_3'] = conv2d(model['relu3_2'], 6)
    model['relu3_3'] = relu(model['conv3_3'])

    model['conv3_4'] = conv2d(model['relu3_3'], 7)
    model['relu3_4'] = relu(model['conv3_4'])

    model['pool3'] = pool2d(model['relu3_4'])

    model['conv4_1'] = conv2d(model['pool3'], 8)
    model['relu4_1'] = relu(model['conv4_1'])

    return model
