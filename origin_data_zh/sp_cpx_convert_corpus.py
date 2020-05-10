import codecs
from filterdata_cla.zh_sim_cplx_convert import *

# convert the examples in complex form of Chinese to simple form

DIRNAME = '/home/shuxiny/home/shuxiny/Tensorflow_NRE_zh/origin_data_zh/'
FILENAME = 'all_sibling'

fin = codecs.open(DIRNAME + FILENAME + '.txt', 'r', 'utf-8')
fout = codecs.open(DIRNAME + FILENAME + '_sim.txt', 'w', 'utf-8')

count = 0
for line in fin:
    count += 1
    if count % 100 == 0:
        print count
    spans = line.strip().split('\t')
    sen = spans[5]
    nsen = convert_cplx2sim(sen.encode('utf8')).decode('utf8')
    if nsen == sen:
        fout.write(line)
    else:
        fout.write(line)
        if len(nsen.split(' ')) != len(sen.split(' ')):
            print 'Error', sen, nsen
            continue
        spans[5] = nsen
        fout.write('\t'.join(spans) + '\n')

fin.close()
fout.close()
