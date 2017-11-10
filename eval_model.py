import tensorflow as tf
import numpy as np
import preprocessing
import TransformNet
import LossFunctions

input_path = 'content/ILSVRC2012_test_00000363.jpeg'



is_training = tf.constant(False, dtype=tf.bool)
input_image = preprocessing.get_resized_image(input_path, 420, 700)
# input_image = input_image - LossFunctions.MEAN_PIXELS
output_image = TransformNet.transform_net(input_image)
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('save/')
    if ckpt and ckpt.model_checkpoint_path:
         saver.restore(sess, ckpt.model_checkpoint_path)
    # sess.run(tf.global_variables_initializer())
    res = sess.run(output_image)

    filename = 'result/res.png'
    filename2 = 'result/res.jpg'
    preprocessing.save_image(filename, res)
    preprocessing.save_image(filename2, res)

