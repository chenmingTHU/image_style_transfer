import tensorflow as tf
import numpy as np
import LossNet
import LossFunctions
import preprocessing
import time


CONTENT_IMAGE_PATH = 'content/chicago.jpg'
STYLE_IMAGE_PATH = 'style/style9.jpg'
CONTENT_IMAGE_PATH2 = 'content/stata.jpg'

IMAGE_SIZE = 256

content_image1 = preprocessing.get_resized_image(CONTENT_IMAGE_PATH, IMAGE_SIZE, IMAGE_SIZE)
content_image1 = content_image1 - LossFunctions.MEAN_PIXELS
content_image2 = preprocessing.get_resized_image(CONTENT_IMAGE_PATH2, IMAGE_SIZE, IMAGE_SIZE)
content_image2 = content_image2 - LossFunctions.MEAN_PIXELS
content_image = tf.concat([content_image1, content_image2], 0)

style_image = preprocessing.get_resized_image(STYLE_IMAGE_PATH, IMAGE_SIZE, IMAGE_SIZE)
style_image = style_image - LossFunctions.MEAN_PIXELS

initial_image = preprocessing.generate_noise_image(content_image, IMAGE_SIZE, IMAGE_SIZE, 1, 0.3)

learning_rate = 5
STEPS = 800

input_image = tf.Variable(np.zeros(shape=[2, IMAGE_SIZE, IMAGE_SIZE, 3]), dtype=tf.float32, trainable=True)
model = LossNet.loss_net(input_image)
TOTAL_LOSS = LossFunctions.total_loss(input_image, content_image, style_image)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(TOTAL_LOSS)

with tf.Session() as sess:
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
            gen_image = gen_image + LossFunctions.MEAN_PIXELS
            print(gen_image.shape)
            print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
            print('   Loss: {:5.1f}'.format(total_loss))
            print('   Time: {}'.format(time.time() - start_time))
            # 计算时间
            start_time = time.time()
            filename = 'result/%d.png' % (index)
            preprocessing.save_image(filename, gen_image)

