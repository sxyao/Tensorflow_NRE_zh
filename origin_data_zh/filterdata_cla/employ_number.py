# coding: utf-8
from zh_sim_cplx_convert import convert_sim2cplx
import codecs

numunit = {'十', '百', '千', '万', '亿'}
timeunit = {'时', '分', '年', '周', '月', '天'}

cplunit = set()
for item in numunit:
    cplunit.add(convert_sim2cplx(item).decode('utf8'))
    cplunit.add(item.decode('utf8'))
numunit = cplunit

cplunit1 = set()
for item in timeunit:
    cplunit1.add(convert_sim2cplx(item).decode('utf8'))
    cplunit1.add(item.decode('utf8'))
timeunit = cplunit1


DIRNAME = '/home/shuxiny/home/shuxiny/Tensorflow_NRE_zh/origin_data_zh/literal_data/cla_process/'
file1 = codecs.open(DIRNAME + 'train_literal_long_number_of_employees_members.txt', 'r', 'utf-8')
file2 = codecs.open(DIRNAME + 'train_literal_long_number_of_employees_members_clean1.txt', 'w', 'utf-8')

for line in file1:
    spans = line.strip().split('\t')
    words = spans[5].split(' ')
    numid = int(spans[7])
    if u'法定 工時 將從 明年 起 調 降為 每 週 四十二 小時 , 行政院 >勞工委 員會 初步 估計 , 約 有 百分之六十八 點 四 事 業單 位 應 配合 調整 , 全國' in spans[5]:
        print 'put'
    if numid - 1 >= 0 and words[numid - 1][-1] in numunit:
        continue
    if numid + 1 < len(words) and words[numid + 1][0] in numunit:
        continue
    if numid + 1 < len(words) and words[numid + 1][0] in timeunit:
        continue
    file2.write(line)

file1.close()
file2.close()
