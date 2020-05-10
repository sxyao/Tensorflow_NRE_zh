import numpy as np
import os
import codecs
from origin_data_zh.filterdata_cla.zh_sim_cplx_convert import *


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init():
    print 'reading word embedding data...'
    word2id = {}
    f = open('./origin_data_zh/vec_all.txt')
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)


    print 'reading relation to id'
    relation2id = {}
    f = open('./origin_data_zh/relation2id_all.txt', 'r')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    # length of sentence is 70
    fixlen = 70
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60

    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_sen_offset = {}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    test_objtype = {}
    test_mention = {}

    f = open('/home/shuxiny/TensorFlow-NRE/zh_datagen/KBP/test_kbp_nre.txt', 'r')
    fent = open('./origin_data_zh/ObjPair4test_kbp1.txt', 'w')
    fsenoffset = open('./origin_data_zh/senoffset_kbp1.txt', 'w')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        en1 = content[0]
        en2 = content[1]
        relation = relation2id['NA']
        tup = (en1, en2)
        if tup not in test_mention:
            test_mention[tup] = content[2:4]
        elif len(content[2]) > len(test_mention[tup][0]) and len(content[3]) > len(test_mention[tup][1]):
            test_mention[tup] = content[2:4]


        if tup not in test_sen:
            test_sen[tup] = []
            test_sen_offset[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1

        sentence = content[5].split(' ')

        en1pos = int(content[6])
        en2pos = int(content[7])
        en1type = content[8]
        en2type = content[9]
        test_objtype[tup] = content[8:10]+content[-2:]

        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        # for i in range(min(fixlen, len(sentence))):
        #     word = 0
        #     if sentence[i] not in word2id:
        #         word = word2id['UNK']
        #     else:
        #         word = word2id[sentence[i]]
        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                if convert_cplx2sim(sentence[i]) in word2id:
                    word = word2id[convert_cplx2sim(sentence[i])]
                elif convert_sim2cplx(sentence[i]) in word2id:
                    word = word2id[convert_sim2cplx(sentence[i])]
                elif i == en1pos and en1type in word2id:
                    word = word2id[en1type]
                elif i == en2pos and en2type in word2id:
                    word = word2id[en2type]
                else:
                    word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)
        test_sen_offset[tup].append(', '.join(content[10:13]))

    test_x = []
    test_y = []

    print 'organizing test data'
    f = open('./data/test_q&a_kbp.txt', 'w')
    temp = 0
    for i in test_sen:
        fent.write(' \t'.join(i) + '\t' + '\t'.join(test_objtype[i]) + '\t' + '\t'.join(test_mention[i])+'\n')
        fsenoffset.write('\t'.join(test_sen_offset[i][:1000]) + '\n')
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./data/testall_x_kbp.npy', test_x)
    np.save('./data/testall_y_kbp.npy', test_y)


def seperate():
    test_word = []
    test_pos1 = []
    test_pos2 = []

    print 'seperating test all data'
    x_test = np.load('./data/testall_x_kbp.npy')

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/testall_word_kbp.npy', test_word)
    np.save('./data/testall_pos1_kbp.npy', test_pos1)
    np.save('./data/testall_pos2_kbp.npy', test_pos2)


def getsmall():
    print 'reading testing data'
    word = np.load('./data/testall_word_kbp.npy')
    pos1 = np.load('./data/testall_pos1_kbp.npy')
    pos2 = np.load('./data/testall_pos2_kbp.npy')
    y = np.load('./data/testall_y_kbp.npy')

    new_word = []
    new_pos1 = []
    new_pos2 = []
    new_y = []

    # we slice some big batch in train data into small batches in case of running out of memory
    print 'get small training data'
    for i in range(len(word)):
        lenth = len(word[i])
        if lenth > 1000:
            new_word.append(word[i][:1000])
            new_pos1.append(pos1[i][:1000])
            new_pos2.append(pos2[i][:1000])
            new_y.append(y[i])
        else:
            new_word.append(word[i])
            new_pos1.append(pos1[i])
            new_pos2.append(pos2[i])
            new_y.append(y[i])

    new_word = np.array(new_word)
    new_pos1 = np.array(new_pos1)
    new_pos2 = np.array(new_pos2)
    new_y = np.array(new_y)

    np.save('./data/small_word_kbp.npy', new_word)
    np.save('./data/small_pos1_kbp.npy', new_pos1)
    np.save('./data/small_pos2_kbp.npy', new_pos2)
    np.save('./data/small_y_kbp.npy', new_y)


init()
seperate()
getsmall()
