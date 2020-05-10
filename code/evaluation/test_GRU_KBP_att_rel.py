import tensorflow as tf
import numpy as np
import time
import pickle
import network_att_rel as network
from sklearn.metrics import average_precision_score

FLAGS = tf.app.flags.FLAGS
# change the name to who you want to send
# tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
tf.app.flags.DEFINE_string('wechat_name', 'filehelper', 'the user you want to send info to')

# if you want to try itchat, please set it to True
itchat_run = False
if itchat_run:
    import itchat


def main(_):
    # ATTENTION: change pathname before you load your model
    pathname = "./model/ATT_GRU_model_ner1_ep7-"

    wordembedding = np.load('./data/vec1.npy')

    test_settings = network.Settings()
    # test_settings.vocab_size = 1163641
    # test_settings.num_classes = 17
    # # test_settings.big_num = 262 * 9
    test_settings.big_num = 10

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                total_y = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    tem1 = np.array(range(0, test_settings.num_steps * len(word_batch[i])))
                    total_y += list(tem1 * test_settings.num_classes + np.argmax(y_batch[i]) + len(total_y))
                    # [[np.argmax(y_batch[i])]]*len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch
                feed_dict[mtest.input_y_expanded] = total_y


                loss, accuracy, prob, prov_values, prov_indices = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob, mtest.prov_values, mtest.prov_indices], feed_dict)
                return prob, prov_values, prov_indices, accuracy

            test_y = np.load('./data/small_y_kbp.npy')
            test_word = np.load('./data/small_word_kbp.npy')
            test_pos1 = np.load('./data/small_pos1_kbp.npy')
            test_pos2 = np.load('./data/small_pos2_kbp.npy')

            for i in xrange(100, 0, -1):
                if len(test_word) % i == 0:
                    test_settings.big_num = i
                    break

            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            saver = tf.train.Saver()

            # ATTENTION: change the list to the iters you want to test !!
            model_iter = 10000

            saver.restore(sess, pathname + str(model_iter))
            print 'Testing all KBP data...'

            allprob = []
            acc = []
            provli_value = []
            provli_index = []
            total_batch = int(len(test_word) / float(test_settings.big_num))
            for i in range(total_batch):
                print i, '/', total_batch
                prob, prov_values, prov_indices, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                provli_value += prov_values
                provli_index += prov_indices

                # acc.append(np.mean(np.reshape(np.array(accuracy), test_settings.big_num)))
                prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                for single_prob in prob:
                    allprob.append(single_prob[1:])

            allprob = np.array(allprob)
            np.save('./out/prob_iter_' + str(model_iter) + '_kbp.npy', allprob)
            allprob = np.reshape(allprob, (-1))

            print 'saving all test result...'

            # ATTENTION: change the save path before you save your result !!
            np.save('./out/allprob_iter_' + str(model_iter) + '_kbp.npy', allprob)

            with open('./out/prov_value_iter_' + str(model_iter) + '_kbp', 'wb') as fp:
                pickle.dump(provli_value, fp)
            with open('./out/prov_index_iter_' + str(model_iter) + '_kbp', 'wb') as fp:
                pickle.dump(provli_index, fp)


if __name__ == "__main__":
    if itchat_run:
        itchat.auto_login(hotReload=True, enableCmdQR=2)
    tf.app.run()
