import tensorflow as tf
import encoder
import numpy as np


MEAN_PIXELS = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 1, 3))
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
CONTENT_LAYER = 'relu4_1'


def content_loss(content_maps, target_maps):

    content_losses = tf.reduce_sum(tf.reduce_mean(tf.square(content_maps - target_maps), axis=[1, 2]))

    return content_losses


def calculate_total_variation_loss(layer):

    shape = tf.shape(layer)
    with tf.Session() as sess:
        shape = sess.run(shape)

    height = shape[1]
    width = shape[2]

    x = tf.slice(layer, begin=[0, 0, 0, 0], size=[-1, height - 1, -1, -1]) \
        - tf.slice(layer, begin=[0, 1, 0, 0], size=[-1, -1, -1, -1])
    y = tf.slice(layer, begin=[0, 0, 0, 0], size=[-1, -1, width - 1, -1]) \
        - tf.slice(layer, begin=[0, 0, 1, 0], size=[-1, -1, -1, -1])

    tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y)/tf.to_float(tf.size(y))

    return tv_loss


def statistics_single_layer_style_loss(input_maps, style_maps):

    eplison = 1e-5

    input_mean, input_var = tf.nn.moments(input_maps, [1, 2], keep_dims=True)
    style_mean, style_var = tf.nn.moments(style_maps, [1, 2], keep_dims=True)

    input_sigma = tf.sqrt(input_var + eplison)
    style_sigma = tf.sqrt(style_var + eplison)

    single_loss_mean = tf.reduce_sum(tf.square(input_mean - style_mean))
    single_loss_var = tf.reduce_sum(tf.square(input_sigma - style_sigma))
    single_loss = single_loss_mean + single_loss_var

    return single_loss


def style_loss(generated_model, style_model):

    style_losses_list = []

    for i in range(len(STYLE_LAYERS)):
        single_loss = statistics_single_layer_style_loss(generated_model[STYLE_LAYERS[i]], style_model[STYLE_LAYERS[i]])
        style_losses_list.append(single_loss)
    style_losses = tf.reduce_sum(style_losses_list)

    return style_losses

