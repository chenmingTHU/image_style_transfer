import tensorflow as tf
import encoder
import decoder
import preprocessing
import AdaIN
import loss
import sys



def eval_arbitrary(content_path, style_path, output_path, height = 560, width = 800):

    # content_name = '002.jpg'
    # style_name = 'style2.jpg'
    # content_path = 'content_test/' + content_name
    # style_path = 'style_test/' + style_name

    content_image = preprocessing.get_resized_image(content_path, height, width)
    style_image = preprocessing.get_resized_image(style_path, height, width)

    content_model = encoder.encoder(content_image - loss.MEAN_PIXELS)
    style_model = encoder.encoder(style_image - loss.MEAN_PIXELS)

    content_maps = content_model['relu4_1']
    style_maps = style_model['relu4_1']

    fusion_maps = AdaIN.adaIn(content_maps, style_maps)

    generated_batches = decoder.decoder(fusion_maps) + loss.MEAN_PIXELS

    saver = tf.train.Saver()

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state('save/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        res = sess.run(generated_batches)
        preprocessing.save_image(output_path, res)

if __name__ == '__main__':
    #eval_arbitrary(content_path, style_path, output_path, height=560, width=800)
    eval_arbitrary(sys.argv[1], sys.argv[2], sys.argv[3], 560, 800)
    print("Finished.")
