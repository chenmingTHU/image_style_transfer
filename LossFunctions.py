import tensorflow as tf
import numpy as np
import LossNet


MEAN_PIXELS = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 1, 3))
# W = [0.1, 0.5, 2, 5]
W = [0.25, 0.25, 0.25, 0.25]
STYLE_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
CONTENT_LAYER = 'relu2_2'

CONTENT_WEIGHT = 0.01
STYLE_WEIGHT = 1
TV_WEIGHT = 0.0001

def calculate_content_loss(content_maps, target_maps):

    shape = tf.shape(target_maps)
    with tf.Session() as sess:
        shape = sess.run(shape)
    size = shape[1] * shape[2] * shape[3]
    batch_size = shape[0]

    content_loss = 0
    for sample in range(batch_size):
        single_content_loss = tf.reduce_sum((content_maps[sample] - target_maps[sample]) ** 2) / size
        content_loss += single_content_loss
    content_loss = content_loss / batch_size

    return content_loss

def calculate_gram_matrix(F):

    shape = tf.shape(F)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    f_reshape = tf.reshape(F, tf.stack([batch_size, height * width, channels]))
    gram = tf.matmul(f_reshape, f_reshape, transpose_a=True) / (tf.to_float(height * width * channels))
    return gram

def calculate_single_layer_style_loss(style_maps, target_maps):

    gram_style = calculate_gram_matrix(style_maps)
    gram_target = calculate_gram_matrix(target_maps)
    batch_size = tf.shape(target_maps)[0]

    with tf.Session() as sess:
        batch_size = sess.run(batch_size)

    style_loss = 0
    for sample in range(batch_size):
        single_style_loss = tf.reduce_sum((gram_style[0] - gram_target[sample]) ** 2)
        style_loss += single_style_loss

    style_loss = style_loss / batch_size
    return style_loss

def calculate_style_loss(style_maps, target_maps):
    num_layer = len(W)
    E = [calculate_single_layer_style_loss(style_maps[i], target_maps[i]) for i in range(num_layer)]

    style_loss = tf.reduce_sum([W[i]*E[i] for i in range(num_layer)])
    return style_loss

# def calculate_style_loss(S, model):
#     num_style_layers = len(STYLE_LAYERS)
#     E = [calculate_single_layer_style_loss(S[i], model[STYLE_LAYERS[i]]) for i in range(num_style_layers)]
#     style_loss = sum([E[i]*W[i] for i in range(num_style_layers)])
#     return style_loss

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



def total_loss(input_image, content_image, style_image):

    model_input = LossNet.loss_net(input_image)
    model_content = LossNet.loss_net(content_image)
    model_style = LossNet.loss_net(style_image)

    content_loss = calculate_content_loss(model_content[CONTENT_LAYER], model_input[CONTENT_LAYER])

    style_maps = [model_style[layer] for layer in STYLE_LAYERS]
    style_maps_input = [model_input[layer] for layer in STYLE_LAYERS]
    style_loss = calculate_style_loss(style_maps, style_maps_input)

    tv_loss = calculate_total_variation_loss(model_input[CONTENT_LAYER])

    loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss
    return loss



