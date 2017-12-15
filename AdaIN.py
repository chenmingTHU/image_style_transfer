import tensorflow as tf

def adaIn(content_maps, style_maps):

    epsilon = 1e-5
    mean_content, var_content = tf.nn.moments(content_maps, [1, 2], keep_dims=True)
    mean_style, var_style = tf.nn.moments(style_maps, [1, 2], keep_dims=True)

    sigma_content = tf.sqrt(var_content + epsilon)
    sigma_style = tf.sqrt(var_style + epsilon)

    res = (content_maps - mean_content) * sigma_style / sigma_content + mean_style
    return res




