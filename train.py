import tensorflow as tf
import numpy as np
import LossNet
import LossFunctions
import read_data
import preprocessing
import TransformNet
import os.path

def train():

    image_path = 'content/'
    is_training = tf.constant(True, dtype=tf.bool)
    STYLE_IMAGE_PATH = 'style/' + 'style4.jpg'
    IMAGE_SIZE = preprocessing.IMAGE_SIZE

    style_image = preprocessing.get_resized_image(STYLE_IMAGE_PATH, IMAGE_SIZE, IMAGE_SIZE)

    image_batchs = read_data.get_batch(image_path=image_path, batch_size=2, capacity=8)
    # image_batchs = preprocessing.generate_noise_image(image_batchs, IMAGE_SIZE, IMAGE_SIZE, 4, noise_ratio=0.3)
    input_data = TransformNet.transform_net(image_batchs)

    loss = LossFunctions.total_loss(input_image=input_data - LossFunctions.MEAN_PIXELS,
                                    content_image=image_batchs - LossFunctions.MEAN_PIXELS,
                                    style_image=style_image - LossFunctions.MEAN_PIXELS)

    learning_rate = 0.001
    STEPS = 40000
    epoch = 2

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        initial_step = 0

        ckpt = tf.train.get_checkpoint_state('save/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

        try:
            for _ in range(epoch):
                for i in range(initial_step, STEPS):
                    if coord.should_stop():
                        break
                    sess.run(train_step)
                    if i % 100 == 0:
                        temp_loss = sess.run(loss)
                        print('Step {}   Loss: {:5.1f}'.format(i, temp_loss))

                    if i % 400 == 0:
                        saver.save(sess, 'save/model_%.5d.ckpt' % i, global_step=i)
        except tf.errors.OutOfRangeError:
            print('DONE!')
        finally:
            coord.request_stop()
        coord.join(threads)

train()

#
#
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i<2:
#
#             losses = sess.run(loss)
#             print(losses)
#
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)