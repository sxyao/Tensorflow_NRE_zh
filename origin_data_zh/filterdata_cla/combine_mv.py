# -*- coding: utf-8 -*-
import codecs

DIRNAME = '/home/shuxiny/home/shuxiny/Tensorflow_NRE_zh/origin_data_zh/obj_data/cla_process/'
file1 = codecs.open(DIRNAME + 'train_long_all_top_members_employees_good.txt', 'r', 'utf-8')
file2 = codecs.open(DIRNAME + 'train_long_all_top_members_employees_good_converted.txt', 'w', 'utf-8')

fout = codecs.open()
for line in file1:
    line = line.strip()
    line = line.split('\t')
    if 'PERSON' not in line[8:10]:
        continue
    line[0],line[1] = line[1], line[0]
    line[2],line[3] = line[3], line[2]
    line[4] = 'per:employee_or_member_of'
    line[7], line[6] = line[6], line[7]
    line[8], line[9] = line[9], line[8]
    file2.write('\t'.join(line)+'\n')


# religion= set([])
# for line in file1:
#     line = line.strip()
#     line = line.split('\t')
#     religion.add(line[1])
#     religion.add(line[3])
#     if line[8] != 'PERSON':
#         continue
#     file2.write('\t'.join(line)+'\n')
#
# print 'religion:'
# for ele in religion:
#     print ele
file1.close()
file2.close()


