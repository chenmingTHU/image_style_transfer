import tensorflow as tf
from tensorflow.python.training import moving_averages

def conv2d(x, input_channel, output_channel, kernel_size, stride, mode='REFLECT'):
    with tf.variable_scope('conv'):

        shape = [kernel_size, kernel_size, input_channel, output_channel]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel_size / 2), int(kernel_size / 2)], [int(kernel_size / 2), int(kernel_size / 2)], [0, 0]],
                          mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, stride, stride, 1], padding='VALID', name='conv')


def conv2d_transpose(input_tensor, input_channel, output_channel, kernel_size, stride, padding_mode='SAME'):

    with tf.variable_scope('conv2d_transpose'):
        weight = tf.Variable(tf.truncated_normal(shape=tf.stack([kernel_size, kernel_size, output_channel, input_channel]),
                                                 stddev=1), dtype=tf.float32, name='weight')

        strides = [1, stride, stride, 1]
        shape = tf.shape(input_tensor)
        with tf.Session() as sess:
            shape = sess.run(shape)
        batch_size = shape[0]
        height = shape[1] * stride
        width = shape[2] * stride
        output_shape = tf.stack([batch_size, height, width, output_channel])

        deconv = tf.nn.conv2d_transpose(input_tensor, weight, output_shape=output_shape,
                                        strides=strides, padding=padding_mode, name='deconv2d')
        return deconv


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


def relu(input_tensor):
    with tf.variable_scope('relu'):
        relu_tmp = tf.nn.relu(input_tensor)
        # res = tf.where(tf.equal(relu_tmp, relu_tmp), relu_tmp, tf.zeros_like(relu_tmp))
        return relu_tmp


def instance_norm(input_tensor):
    epsilon = 1e-9
    mean, var = tf.nn.moments(input_tensor, [1, 2], keep_dims=True)
    res = tf.div(tf.subtract(input_tensor, mean), tf.sqrt(tf.add(var, epsilon)))
    return res


def batch_norm(input_tensor, is_training, eps=0.0001, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='batch_norm'):
        shape = tf.shape(input_tensor)
        with tf.Session() as sess:
            shape = sess.run(shape)
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



def residual(input_tensor, input_channel, kernel_size=3, stride=1):

    # the input tensor and the output tensor must have the same shape
    with tf.variable_scope('residual'):
        conv_1 = conv2d(input_tensor, input_channel, input_channel, kernel_size, stride)
        bn_1 = instance_norm(conv_1)
        relu_1 = relu(bn_1)
        conv_2 = conv2d(relu_1, input_channel, input_channel, kernel_size, stride)
        bn_2 = instance_norm(conv_2)

        res = bn_2 + input_tensor
        return res


def transform_net(input_image, tarining=True):

    input_image = tf.pad(input_image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv_1'):
        conv_1 = conv2d(input_image, input_channel=3, output_channel=32, kernel_size=9, stride=1)
        bn_1 = instance_norm(conv_1)
        relu_1 = relu(bn_1)
    with tf.variable_scope('conv_2'):
        conv_2 = conv2d(relu_1, input_channel=32, output_channel=64, kernel_size=3, stride=2)
        bn_2 = instance_norm(conv_2)
        relu_2 = relu(bn_2)
    with tf.variable_scope('conv_3'):
        conv_3 = conv2d(relu_2, input_channel=64, output_channel=128, kernel_size=3, stride=2)
        bn_3 = instance_norm(conv_3)
        relu_3 = relu(bn_3)
    with tf.variable_scope('residual_1'):
        residual1 = residual(relu_3, 128)
    with tf.variable_scope('residual_2'):
        residual2 = residual(residual1, 128)
    with tf.variable_scope('residual_3'):
        residual3 = residual(residual2, 128)
    with tf.variable_scope('residual_4'):
        residual4 = residual(residual3, 128)
    with tf.variable_scope('residual_5'):
        residual5 = residual(residual4, 128)
    with tf.variable_scope('deconv_1'):
        deconv_1 = resize_conv2d(residual5, 128, 64, 3, 1, tarining)
        d_bn_1 = instance_norm(deconv_1)
        d_relu_1 = relu(d_bn_1)
    with tf.variable_scope('deconv_2'):
        deconv_2 = resize_conv2d(d_relu_1, 64, 32, 3, 1, tarining)
        d_bn_2 = instance_norm(deconv_2)
        d_relu_2 = relu(d_bn_2)
    with tf.variable_scope('deconv_3'):
        deconv_3 = conv2d(d_relu_2, 32, 3, 9, 1)
        d_bn_3 = instance_norm(deconv_3)
        res = tf.nn.tanh(d_bn_3)

    y = (res + 1) * 127.5

    shape_y = tf.shape(y)
    with tf.Session() as sess:
        shape_y = sess.run(shape_y)
    height = shape_y[1]
    width = shape_y[2]
    batch_size = shape_y[0]
    channels = shape_y[3]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([batch_size, height - 20, width - 20, channels]))
    return y


