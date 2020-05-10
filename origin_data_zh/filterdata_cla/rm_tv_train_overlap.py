import codecs
import pickle

fin = codecs.open('../train_clean_sim1_noredun.txt', 'r', 'utf-8')
fout = codecs.open('../tv_clean_sim1_noredun_notrain.txt', 'w', 'utf-8')

ent_dic = set()
for line in fin:
    if line.strip() == '':
        continue
    spans = line.split('\t')
    ent_dic.add(tuple(spans[0:2]+[spans[4]]))

fin2 = codecs.open('../tv_clean_sim1_noredun.txt', 'r', 'utf-8')
for line in fin2:
    if line.strip() == '':
        continue
    spans = line.split('\t')
    if tuple(spans[0:2]+[spans[4]]) not in ent_dic:
        fout.write(line)

fin.close()
fout.close()
fin2.close()


