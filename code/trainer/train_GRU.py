import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_prov1 as network
from tensorflow.contrib.tensorboard.plugins import projector
from collections import defaultdict

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')

# change the name to who you want to send
# tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
tf.app.flags.DEFINE_string('wechat_name', 'filehelper', 'the user you want to send info to')

# if you want to try itchat, please set it to True
itchat_run = False
if itchat_run:
    import itchat


def oversample(train_y, train_word, train_pos1, train_pos2):
    dic_word = defaultdict(list)
    for i in xrange(len(train_y)):
        dic_word[tuple(train_y[i])].append(i)

    # find the uniques labels for each class and count the number of instances for each class
    class_map = defaultdict(list)
    class_total = [0] * len(train_y[0])
    for k in dic_word:
        inds = np.where(np.array(k) == 1)[0]
        for ele in inds:
            class_map[ele].append(k)
            class_total[ele] += len(dic_word[k])
    print class_total

    # put sentences with fewer label at front
    for k in class_map:
        class_map[k] = sorted(class_map[k], key=lambda x: np.sum(np.array(x)))

    sorted_class = sorted(range(1, len(train_y[0])), key=lambda x: class_total[x], reverse=True)
    maxnum = class_total[sorted_class[0]]
    cla = 0
    if class_total[cla] > maxnum:
        newins = np.random.choice(dic_word[class_map[cla][0]], size=maxnum*8, replace=True)
        dic_word[class_map[cla][0]] = list(newins)

    for ind in xrange(0, len(sorted_class)):
        cla = sorted_class[ind]
        if class_total[cla] == maxnum:
            continue
        new_class_total = 0
        for ele in class_map[cla]:
            new_class_total += len(dic_word[ele])
        if new_class_total >= maxnum:
            continue
        new_instance = maxnum - new_class_total
        try:
            newins = np.random.choice(dic_word[class_map[cla][0]], size=new_instance, replace=True)
        except:
            continue
        dic_word[class_map[cla][0]] += list(newins)

    n_train_y, n_train_word, n_train_pos1, n_train_pos2 = np.array([]), np.array([]), np.array([]), np.array([])

    for k in dic_word:
        if len(n_train_y)==0:
            n_train_y = train_y[dic_word[k]]
            n_train_word = train_word[dic_word[k]]
            n_train_pos1 = train_pos1[dic_word[k]]
            n_train_pos2 = train_pos2[dic_word[k]]
            continue
        n_train_y = np.concatenate((n_train_y, train_y[dic_word[k]]), axis=0)
        n_train_word = np.concatenate((n_train_word, train_word[dic_word[k]]), axis=0)
        n_train_pos1 = np.concatenate((n_train_pos1, train_pos1[dic_word[k]]), axis=0)
        n_train_pos2 = np.concatenate((n_train_pos2, train_pos2[dic_word[k]]), axis=0)

    return n_train_y, n_train_word, n_train_pos1, n_train_pos2


def main(_):
    # the path to save models
    save_path = './model/'

    print 'reading wordembedding'
    wordembedding = np.load('./data/vec1_newtvsplit.npy')

    print 'reading training data'
    train_y = np.load('./data/small_y_newtvsplit.npy')
    train_word = np.load('./data/small_word_newtvsplit.npy')
    train_pos1 = np.load('./data/small_pos1_newtvsplit.npy')
    train_pos2 = np.load('./data/small_pos2_newtvsplit.npy')
    train_y, train_word, train_pos1, train_pos2 = oversample(train_y, train_word, train_pos1, train_pos2)

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])

    big_num = settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)

            # train_op=optimizer.minimize(m.total_loss,global_step=global_step)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(max_to_keep=None)

            # merged_summary = tf.summary.merge_all()
            merged_summary = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            # summary for embedding
            # it's not available in tf 0.11,(because there is no embedding panel in 0.11's tensorboard) so I delete it =.=
            # you can try it on 0.12 or higher versions but maybe you should change some function name at first.

            # summary_embed_writer = tf.train.SummaryWriter('./model',sess.graph)
            # config = projector.ProjectorConfig()
            # embedding_conf = config.embedding.add()
            # embedding_conf.tensor_name = 'word_embedding'
            # embedding_conf.metadata_path = './data/metadata.tsv'
            # projector.visualize_embeddings(summary_embed_writer, config)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
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

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 50 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    print(tempstr)
                    if itchat_run:
                        itchat.send(tempstr, FLAGS.wechat_name)

            for one_epoch in range(settings.num_epochs):
                if itchat_run:
                    itchat.send('epoch ' + str(one_epoch) + ' starts!', FLAGS.wechat_name)

                temp_order = range(len(train_word))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        print 'out of range'
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step >= 7000 and current_step % 500 == 0:
                        # if current_step == 50:
                        print 'saving model'
                        path = saver.save(sess, save_path + 'ATT_GRU_final_', global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print tempstr

            if itchat_run:
                itchat.send('training has been finished!', FLAGS.wechat_name)


if __name__ == "__main__":
    if itchat_run:
        itchat.auto_login(hotReload=True, enableCmdQR=2)
    tf.app.run()
