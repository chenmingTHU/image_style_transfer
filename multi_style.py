from __future__ import print_function
import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageOps
import scipy.misc



vgg_model_path = 'imagenet-vgg-verydeep-19.mat'

# vgg19 = scipy.io.loadmat(vgg_model_path)
# vgg19_layers = vgg19['layers']

def get_resized_image(img_path, height, width):
    image = Image.open(img_path)
    # it's because PIL is column major so you have to change place of width and height
    # this is stupid, i know
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def generate_noise_image(content_image, height, width, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20,
                                    (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)




def get_weights(vgg19_layers, layer_index, expected_layer_name):
    W = vgg19_layers[0][layer_index][0][0][0][0][0]
    b = vgg19_layers[0][layer_index][0][0][0][0][1]
    layer_name = vgg19_layers[0][layer_index][0][0][3]

    assert layer_name == expected_layer_name
    return W, b

# W, b = get_weights(vgg19_layers, 0, 'conv1_1')

def conv2d_relu(vgg19_layers, pre_layer, layer_index, layer_name):
    W_get, b_get = get_weights(vgg19_layers=vgg19_layers, layer_index=layer_index, expected_layer_name=layer_name)
    W = tf.constant(W_get, dtype=tf.float32)
    b = tf.constant(b_get, dtype=tf.float32)

    conv2d = tf.nn.conv2d(pre_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')

    return tf.nn.relu(conv2d + b)

def avgpooling(pre_layer):

    return tf.nn.avg_pool(pre_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='avg_pool')

def get_model_outputs(path, input_image):

    vgg19 = scipy.io.loadmat(path)
    vgg19_layers = vgg19['layers']

    model = {}
    model['conv1_1'] = conv2d_relu(vgg19_layers, input_image, 0, 'conv1_1')
    model['conv1_2'] = conv2d_relu(vgg19_layers, model['conv1_1'], 2, 'conv1_2')
    model['avgpool1'] = avgpooling(model['conv1_2'])

    model['conv2_1'] = conv2d_relu(vgg19_layers, model['avgpool1'], 5, 'conv2_1')
    model['conv2_2'] = conv2d_relu(vgg19_layers, model['conv2_1'], 7, 'conv2_2')
    model['avgpool2'] = avgpooling(model['conv2_2'])

    model['conv3_1'] = conv2d_relu(vgg19_layers, model['avgpool2'], 10, 'conv3_1')
    model['conv3_2'] = conv2d_relu(vgg19_layers, model['conv3_1'], 12, 'conv3_2')
    model['conv3_3'] = conv2d_relu(vgg19_layers, model['conv3_2'], 14, 'conv3_3')
    model['conv3_4'] = conv2d_relu(vgg19_layers, model['conv3_3'], 16, 'conv3_4')
    model['avgpool3'] = avgpooling(model['conv3_4'])

    model['conv4_1'] = conv2d_relu(vgg19_layers, model['avgpool3'], 19, 'conv4_1')
    model['conv4_2'] = conv2d_relu(vgg19_layers, model['conv4_1'], 21, 'conv4_2')
    model['conv4_3'] = conv2d_relu(vgg19_layers, model['conv4_2'], 23, 'conv4_3')
    model['conv4_4'] = conv2d_relu(vgg19_layers, model['conv4_3'], 25, 'conv4_4')
    model['avgpool4'] = avgpooling(model['conv4_4'])

    model['conv5_1'] = conv2d_relu(vgg19_layers, model['avgpool4'], 28, 'conv5_1')
    model['conv5_2'] = conv2d_relu(vgg19_layers, model['conv5_1'], 30, 'conv5_2')
    model['conv5_3'] = conv2d_relu(vgg19_layers, model['conv5_2'], 32, 'conv5_3')
    model['conv5_4'] = conv2d_relu(vgg19_layers, model['conv5_3'], 34, 'conv5_4')
    model['avgpool5'] = avgpooling(model['conv5_4'])

    return model

# input_image = np.random.normal(0, 10, size=(1, 400, 400, 3)).astype(np.float32)
# model = get_model_outputs(vgg_model_path, input_image)
# print(model)

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
weights_style = [0.5, 1, 2, 4, 10]
CONTENT_LAYER = 'conv5_1'
# CONTENT_LAYER = 'conv1_1'

STYLE_WEIGHT = 1
CONTENT_WEIGHT = 0.001
# STYLE_WEIGHT = 1
# CONTENT_WEIGHT = 0.1

STYLE_IMAGE_PATH = ['style/style19.jpg', 'style/style2.jpg']
multi_style_weight = [0.5, 0.5]
CONTENT_IMAGE_PATH = 'content/chicago.jpg'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def calculate_content_loss(p, f):
    content_loss = tf.reduce_sum((p - f)**2) / 2
    # content_loss = tf.reduce_sum((p - f)**2) / (4*p.size)
    return content_loss

def calculate_gram_matrix(F, N, M):
    # N is the number of feature maps
    # M is the size of a single map

    F_reshape = tf.reshape(F, shape=(M, N))

    gram = tf.matmul(tf.transpose(F_reshape), F_reshape)
    return gram

def calculate_single_layer_style_loss(s, f):

    N_s = s.shape[3]
    M_s = s.shape[1]*s.shape[2]


    gram_s = calculate_gram_matrix(s, N_s, M_s)

    gram_f = calculate_gram_matrix(f, N_s, M_s)

    single_style_loss = tf.reduce_sum((gram_s - gram_f)**2) / ((2*N_s*N_s)**2)
    return single_style_loss

def calculate_style_loss(S, model):
    num_style_layers = len(STYLE_LAYERS)

    E = [calculate_single_layer_style_loss(S[i], model[STYLE_LAYERS[i]]) for i in range(num_style_layers)]

    style_loss = sum([E[i]*weights_style[i] for i in range(num_style_layers)])

    return style_loss

def calculate_total_loss(model, input_image, style_image, content_image):

    num_style = len(multi_style_weight)
    style_loss = 0
    with tf.Session() as sess:
        sess.run(input_image.assign(content_image))
        con_p = sess.run(model[CONTENT_LAYER])

    content_loss = calculate_content_loss(con_p, model[CONTENT_LAYER])
    with tf.Session() as sess:
        for style in range(num_style):
            sess.run(input_image.assign(style_image[style]))
            sty_p = sess.run([model[layers] for layers in STYLE_LAYERS])
            style_loss += multi_style_weight[style] * calculate_style_loss(sty_p, model)


    total_loss = content_loss*CONTENT_WEIGHT + style_loss*STYLE_WEIGHT

    return total_loss


input_image = tf.Variable(np.zeros(shape=[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32, trainable=True)

model = get_model_outputs(vgg_model_path, input_image)

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

content_image = get_resized_image(CONTENT_IMAGE_PATH, IMAGE_HEIGHT, IMAGE_WIDTH)
content_image = content_image - MEAN_PIXELS
style_image = []
num_style = len(multi_style_weight)

for style in range(num_style):
    temp_style_image = get_resized_image(STYLE_IMAGE_PATH[style], IMAGE_HEIGHT, IMAGE_WIDTH)
    temp_style_image = temp_style_image - MEAN_PIXELS
    style_image.append(temp_style_image)




learning_rate = 5
STEPS = 800

TOTAL_LOSS = calculate_total_loss(model, input_image, style_image, content_image)

optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(TOTAL_LOSS)

initial_image = generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, noise_ratio=0.6)
# initial_image = generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, noise_ratio=1)


with tf.Session() as sess:
    # 初始化变量创建保存器和summary的writer
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    skip_step = 1

    sess.run(input_image.assign(initial_image))
    start_time = time.time()
    for index in range(STEPS):

        if index >= 5 and index < 20:
            skip_step = 10
        elif index >= 20:
            skip_step = 20
        sess.run(optimizer)

        if (index + 1) % skip_step == 0:
            gen_image, total_loss = sess.run([input_image, TOTAL_LOSS])
            gen_image = gen_image + MEAN_PIXELS
            print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
            print('   Loss: {:5.1f}'.format(total_loss))
            print('   Time: {}'.format(time.time() - start_time))
            # 计算时间
            start_time = time.time()
            filename = 'outputs/%d.png' % (index)
            save_image(filename, gen_image)
            # if (index + 1) % 20 == 0:
            #     gen_image = gen_image.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            #     plt.figure(1)
            #     plt.imshow(gen_image)
            #     plt.show()



