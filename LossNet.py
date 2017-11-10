import tensorflow as tf
import scipy.io


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


def loss_net(input_image):

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




