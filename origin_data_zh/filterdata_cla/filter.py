# -*- coding: utf-8 -*-
import codecs

DIRNAME = '/home/shuxiny/home/shuxiny/Tensorflow_NRE_zh/origin_data_zh/obj_data/cla_process/'
file1 = codecs.open(DIRNAME + 'train_long_all_religion.txt', 'r', 'utf-8')
file2 = codecs.open(DIRNAME + 'train_long_all_political_religious_affiliation.txt', 'w', 'utf-8')
file3 = codecs.open(DIRNAME + 'train_long_all_religion.txt', 'w', 'utf-8')
org = set()
orgO = set()
orgPER = set()
lines = file1.read().splitlines()

for line in lines:
    line = line.strip()
    line = line.split('\t')
    print line[0], line[8]
    if line[8] == 'ORGANIZATION' or line[0] == u'公教中学':
        org.add(line[0])
    if line[8] == 'O':
        orgO.add(line[0])
    if line[8] == 'PERSON':
        orgPER.add(line[0])

print 'ORGANIZATION', len(org)
for item in org:
    print item

print 'O', len(orgO)
for item in orgO:
    print item

print 'PERSON', len(orgPER)
for item in orgPER:
    print item

orgPER = orgPER.union(orgO)

for line in lines:
    line = line.strip()
    line = line.split('\t')
    if line[0] in orgPER:
        file2.write(line+'\n')
    if line[8] == 'PERSON' or (line[0] in orgPER and line[0][-1] not in {u'人', u'族'}):
        file3.write(line+'\n')


file1.close()
file2.close()
file3.close()