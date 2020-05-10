# coding: utf-8
import numpy as np
import os
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
    vec = []
    word2id = {}
    f = open('./origin_data_zh/vec_all.txt')
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

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

    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

    mapped_count = 0
    mapped_sim = 0
    mapped_cplx = 0
    mapped_type = 0
    no_map = 0

    print 'reading train data...'
    f = open('./origin_data_zh/train_clean_sim1_noredun.txt', 'r')

    while True:
        content = f.readline()
        if content.strip() == '':
            break

        content = content.strip().split('\t')
        # get entity name
        try:
            en1 = content[0]
            en2 = content[1]
        except:
            print 'content', content
            raise IndexError

        relation = 0
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]
        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        label_tag = 0
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[5].split(' ')

        en1pos = int(content[6])
        en2pos = int(content[7])
        en1type = content[8]
        en2type = content[9]

        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                if convert_cplx2sim(sentence[i]) in word2id:
                    word = word2id[convert_cplx2sim(sentence[i])]
                    mapped_sim += 1
                elif convert_sim2cplx(sentence[i]) in word2id:
                    word = word2id[convert_sim2cplx(sentence[i])]
                    mapped_cplx += 1
                elif i == en1pos and en1type in word2id:
                    word = word2id[en1type]
                    mapped_type += 1
                elif i == en2pos and en2type in word2id:
                    word = word2id[en2type]
                    mapped_type += 1
                else:
                    word = word2id['UNK']
                    no_map += 1
            else:
                word = word2id[sentence[i]]
                mapped_count += 1

            output[i][0] = word

        train_sen[tup][label_tag].append(output)

    print 'train_sen', len(train_sen)
    totalcount = float(mapped_sim + mapped_count + mapped_cplx + mapped_type + no_map)
    print 'mapped_count', mapped_count, mapped_count/totalcount
    print 'mapped_sim', mapped_sim, mapped_sim / totalcount
    print 'mapped_cplx', mapped_cplx, mapped_cplx / totalcount
    print 'mapped_type', mapped_type, mapped_type / totalcount
    print 'no_map', no_map, no_map / totalcount


    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

    f = open('./origin_data_zh/tv_clean_sim1_noredun_notrain.txt', 'r')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        en1 = content[0]
        en2 = content[1]

        relation = 0
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]
        tup = (en1, en2)

        if tup not in test_sen:
            test_sen[tup] = []
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

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    print 'organizing train data'
    f = open('./data/train_q&a_long.txt', 'w')
    temp = 0
    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print 'ERROR'
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print 'organizing test data'
    f = open('./data/test_q&a_long.txt', 'w')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./data/vec1.npy', vec)
    np.save('./data/train_x_ner1.npy', train_x)
    np.save('./data/train_y_ner1.npy', train_y)
    np.save('./data/testall_x_ner1.npy', test_x)
    np.save('./data/testall_y_ner1.npy', test_y)

    # get test data for P@N evaluation, in which only entity pairs with more than 1 sentence exist
    print 'get test data for p@n test'

    pone_test_x = []
    pone_test_y = []

    ptwo_test_x = []
    ptwo_test_y = []

    pall_test_x = []
    pall_test_y = []

    for i in range(len(test_x)):
        if len(test_x[i]) > 1:

            pall_test_x.append(test_x[i])
            pall_test_y.append(test_y[i])

            onetest = []
            temp = np.random.randint(len(test_x[i]))
            onetest.append(test_x[i][temp])
            pone_test_x.append(onetest)
            pone_test_y.append(test_y[i])

            twotest = []
            temp1 = np.random.randint(len(test_x[i]))
            temp2 = np.random.randint(len(test_x[i]))
            while temp1 == temp2:
                temp2 = np.random.randint(len(test_x[i]))
            twotest.append(test_x[i][temp1])
            twotest.append(test_x[i][temp2])
            ptwo_test_x.append(twotest)
            ptwo_test_y.append(test_y[i])

    pone_test_x = np.array(pone_test_x)
    pone_test_y = np.array(pone_test_y)
    ptwo_test_x = np.array(ptwo_test_x)
    ptwo_test_y = np.array(ptwo_test_y)
    pall_test_x = np.array(pall_test_x)
    pall_test_y = np.array(pall_test_y)

    np.save('./data/pone_test_x_ner1.npy', pone_test_x)
    np.save('./data/pone_test_y_ner1.npy', pone_test_y)
    np.save('./data/ptwo_test_x_ner1.npy', ptwo_test_x)
    np.save('./data/ptwo_test_y_ner1.npy', ptwo_test_y)
    np.save('./data/pall_test_x_ner1.npy', pall_test_x)
    np.save('./data/pall_test_y_ner1.npy', pall_test_y)


def seperate():
    print 'reading training data'
    x_train = np.load('./data/train_x_ner1.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print 'seprating train data'
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
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
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./data/train_word_ner1.npy', train_word)
    np.save('./data/train_pos1_ner1.npy', train_pos1)
    np.save('./data/train_pos2_ner1.npy', train_pos2)

    print 'reading p-one test data'
    x_test = np.load('./data/pone_test_x_ner1.npy')
    print 'seperating p-one test data'
    test_word = []
    test_pos1 = []
    test_pos2 = []

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
    np.save('./data/pone_test_word_ner1.npy', test_word)
    np.save('./data/pone_test_pos1_ner1.npy', test_pos1)
    np.save('./data/pone_test_pos2_ner1.npy', test_pos2)

    print 'reading p-two test data'
    x_test = np.load('./data/ptwo_test_x_ner1.npy')
    print 'seperating p-two test data'
    test_word = []
    test_pos1 = []
    test_pos2 = []

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
    np.save('./data/ptwo_test_word_ner1.npy', test_word)
    np.save('./data/ptwo_test_pos1_ner1.npy', test_pos1)
    np.save('./data/ptwo_test_pos2_ner1.npy', test_pos2)

    print 'reading p-all test data'
    x_test = np.load('./data/pall_test_x_ner1.npy')
    print 'seperating p-all test data'
    test_word = []
    test_pos1 = []
    test_pos2 = []

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
    np.save('./data/pall_test_word_ner1.npy', test_word)
    np.save('./data/pall_test_pos1_ner1.npy', test_pos1)
    np.save('./data/pall_test_pos2_ner1.npy', test_pos2)

    print 'seperating test all data'
    x_test = np.load('./data/testall_x_ner1.npy')

    test_word = []
    test_pos1 = []
    test_pos2 = []

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

    np.save('./data/testall_word_ner1.npy', test_word)
    np.save('./data/testall_pos1_ner1.npy', test_pos1)
    np.save('./data/testall_pos2_ner1.npy', test_pos2)


def getsmall():
    print 'reading training data'
    word = np.load('./data/train_word_ner1.npy')
    pos1 = np.load('./data/train_pos1_ner1.npy')
    pos2 = np.load('./data/train_pos2_ner1.npy')
    y = np.load('./data/train_y_ner1.npy')

    new_word = []
    new_pos1 = []
    new_pos2 = []
    new_y = []

    # we slice some big batch in train data into small batches in case of running out of memory
    print 'get small training data'
    for i in range(len(word)):
        lenth = len(word[i])
        if lenth > 75:
            for start in xrange(0, min(len(word[i]), 75 * 10), 75):
                new_word.append(word[i][start:start + 75])
                new_pos1.append(pos1[i][start:start + 75])
                new_pos2.append(pos2[i][start:start + 75])
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

    np.save('./data/small_word_ner1.npy', new_word)
    np.save('./data/small_pos1_ner1.npy', new_pos1)
    np.save('./data/small_pos2_ner1.npy', new_pos2)
    np.save('./data/small_y_ner1.npy', new_y)


# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/testall_y_ner1.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./data/allans_ner1.npy', allans)


def get_metadata():
    fwrite = open('./data/metadata.tsv', 'w')
    f = open('./origin_data_zh/vec_all.txt')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]
        fwrite.write(name + '\n')
    f.close()
    fwrite.close()


init()
seperate()
getsmall()
getans()
get_metadata()
