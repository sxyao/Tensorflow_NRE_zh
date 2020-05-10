# -*- coding: utf-8 -*-
import codecs

DIRNAME = '/home/shuxiny/home/shuxiny/Tensorflow_NRE_zh/origin_data_zh/'
file = codecs.open(DIRNAME + 'all_sibling_sim.txt', 'r', 'utf-8')
fileout = codecs.open(DIRNAME + 'all_sibling_sim_noredun.txt', 'w', 'utf-8')

lines = file.read()
lines = lines.split('\n')
setline = set()

for line in lines:
    if line in setline:
        continue
    setline.add(line)
    fileout.write(line+'\n')

file.close()
fileout.close()


