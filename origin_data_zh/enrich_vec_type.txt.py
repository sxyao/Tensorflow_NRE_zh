import numpy as np

fin = open('type2id.txt', 'r')
fout = open('vec_enttype.txt', 'w')

for line in fin:
    spans = line.split('\t')
    vec = list(np.random.random_sample(50) * 16 - 8)
    fout.write(spans[0] + ' ' + ' '.join([str(item) for item in vec]) + '\n')

fin.close()
fout.close()
