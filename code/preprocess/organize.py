# coding: utf-8
import numpy as np
import codecs
import requests
import pickle

# map id to relation
id2relation = {}
f = open('./origin_data_zh/relation2id_all.txt', 'r')
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split()
    id2relation[int(content[1])] = content[0]
f.close()

clfrel2kbprel = {('per:place_of_birth', 'COUNTRY'): 'per:country_of_birth',
                 ('per:place_of_birth', 'STATE_OR_PROVINCE'): 'per:stateorprovince_of_birth',
                 ('per:place_of_birth', 'CITY'): 'per:city_of_birth',
                 ('per:place_of_birth', 'GPE'): 'per:city_of_birth',
                 ('per:place_of_birth', 'LOCATION'): 'per:city_of_birth',
                 ('per:place_of_birth', 'O'): 'per:city_of_birth',

                 ('per:place_of_death', 'COUNTRY'): 'per:country_of_death',
                 ('per:place_of_death', 'STATE_OR_PROVINCE'): 'per:stateorprovince_of_death',
                 ('per:place_of_death', 'CITY'): 'per:city_of_death',
                 ('per:place_of_death', 'GPE'): 'per:city_of_death',
                 ('per:place_of_death', 'LOCATION'): 'per:city_of_death',
                 ('per:place_of_death', 'O'): 'per:city_of_death',

                 ('per:place_of_residence', 'COUNTRY'): 'per:countries_of_residence',
                 ('per:place_of_residence', 'STATE_OR_PROVINCE'): 'per:statesorprovinces_of_residence',
                 ('per:place_of_residence', 'CITY'): 'per:cities_of_residence',
                 ('per:place_of_residence', 'GPE'): 'per:cities_of_residence',
                 ('per:place_of_residence', 'LOCATION'): 'per:cities_of_residence',
                 ('per:place_of_residence', 'O'): 'per:cities_of_residence',

                 ('org:place_of_headquarters', 'COUNTRY'): 'org:country_of_headquarters',
                 ('org:place_of_headquarters', 'STATE_OR_PROVINCE'): 'org:stateorprovince_of_headquarters',
                 ('org:place_of_headquarters', 'CITY'): 'org:city_of_headquarters',
                 ('org:place_of_headquarters', 'GPE'): 'org:city_of_headquarters',
                 ('org:place_of_headquarters', 'LOCATION'): 'org:city_of_headquarters',
                 ('org:place_of_headquarters', 'O'): 'org:city_of_headquarters',
                 }


def getPlaceTypeMappedRel(ori_rel, coreType, mention):
    # method1:
    # return clfrel2kbprel[(ori_rel, coreType)]
    # method2:
    payload = {'name': mention, 'searchlang': 'zh', 'lang': 'zh', 'type': 'json', 'username': 'shuxinyao'}
    r = requests.post('http://api.geonames.org/search?', data=payload)
    r.encoding = 'UTF-8'
    js = r.json()
    entities = js[u'geonames']
    if not entities or len(entities) == 0:
        return clfrel2kbprel[(ori_rel, coreType)]
    entity = entities[0]
    if entity[u'fcl'] == u'A':
        if entity[u'fcode'] == u'PCLI':
            return clfrel2kbprel[(ori_rel, 'COUNTRY')]
        else:
            return clfrel2kbprel[(ori_rel, 'STATE_OR_PROVINCE')]
    elif entity[u'fcl'] == u'P':
        return clfrel2kbprel[(ori_rel, 'CITY')]
    elif coreType in {'COUNTRY', 'STATE_OR_PROVINCE', 'CITY'}:
        return clfrel2kbprel[(ori_rel, coreType)]
    else:
        return 'NA'


relation_type = {'per:children': ('PER', 'PER'),
                 'per:siblings': ('PER', 'PER'),
                 'per:other_family': ('PER', 'PER'),
                 'per:place_of_birth': ('PER', 'GPE'),
                 'per:origin': ('PER', 'GPE'),
                 'per:place_of_death': ('PER', 'GPE'),
                 'per:place_of_residence': ('PER', 'GPE'),
                 'per:schools_attended': ('PER', 'ORG'),
                 'per:cause_of_death': ('PER', 'String'),
                 'per:title': ('PER', 'String', 'TITLE'),
                 'per:employee_or_member_of': ('PER', 'ORG|GPE'),
                 'per:religion': ('PER', 'String'),
                 'per:spouse': ('PER', 'PER'),
                 'org:political_religious_affiliation': ('ORG', 'String'),
                 'org:top_members_employees': ('ORG|GPE', 'PER'),
                 'org:subsidiaries': ('ORG|GPE', 'ORG|GPE'),
                 'org:founded_by': ('ORG|GPE', 'PER|ORG|GPE'),
                 'org:place_of_headquarters': ('ORG', 'GPE'),
                 'per:date_of_birth': ('PER', 'String', 'DATE'),
                 'per:date_of_death': ('PER', 'String', 'DATE'),
                 'org:number_of_employees_members': ('ORG', 'String', 'NUMBER'),
                 'org:date_founded': ('ORG|GPE', 'String', 'DATE'),
                 'org:date_dissolved': ('ORG|GPE', 'String', 'DATE'),
                 'org:website': ('ORG', 'String', 'URL')
                 }


def check_type(types, rel):
    requirement = relation_type[rel]
    if types[2] not in requirement[0].split('|'):
        return False
    if types[3] not in requirement[1].split('|'):
        return False
    if len(requirement) == 3:
        if types[1] not in requirement[2].split('|'):
            return False
    return True


fin1 = codecs.open('origin_data_zh/ObjPair4test_kbp1.txt', 'r', 'utf-8')

fin2 = open('origin_data_zh/senoffset_kbp1.txt', 'r')
offsets = []
for line in fin2:
    curoffsets = line.strip().split('\t')
    curoffsets = [item.split(', ') for item in curoffsets]
    offsets.append(curoffsets)

probs = np.load('./out/prob_iter_11000_kbp.npy')
with open('./out/prov_index_iter_11000_kbp', 'rb') as fp:
    provs = pickle.load(fp)

fout_str = codecs.open('KBP_result/strings_kb.txt', 'w', 'utf-8')
fout = open('KBP_result/kbp_sf_output_nre.txt', 'w')

str_count = 0
count = 0
threshold = np.array([0.80315073, 0.95244459, 0.68519044, 0.91919198, 0.42961965,
                      0.80406384, 0.27171884, 0.82861034, 0.94597292, 0.81297968,
                      0.74270612, 0.05174323, 0.9074896, 0.96386564, 0.99999587,
                      0.92542396, 0.88860277, 0.93397575, 0.94161242, 0.7316041,
                      0.17831313, 0.69824608, 0.46919601])

# [0.74954611, 0.9, 0.49541527, 0.3339825, 0.1258869, 0.9, 0.9, 0.47370049, 0.9, 0.65590525, 0.9, 0.46458086, 0.67842948, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

mentiondic = {}

for line in fin1:
    items = line.strip().split('\t')
    if len(items) == 0:
        continue
    tem = probs[count] >= threshold
    clas = np.where(tem == 1)[0]
    if len(clas) > 0:
        clas = clas + 1
        for ele in clas:
            rel = id2relation[ele]
            # type check
            if not check_type(items[2:6], rel):
                continue
            if rel in {'per:place_of_birth', 'per:place_of_death', 'per:place_of_residence',
                       'org:place_of_headquarters'}:
                try:
                    rel = getPlaceTypeMappedRel(rel, items[3], items[-1])
                except:
                    print 'Error: no map', rel, items[3]
                    continue
                if rel == 'NA':
                    continue

            providli = list(provs[count][ele])

            # providli = providli[0:min(5, len(providli))]
            for provid in providli:
                curprov = offsets[count][provid]
                if items[-3] == 'String':
                    document = curprov[0].split(':')
                    curstr = [':String_' + str(str_count), items[1], document[0] + ':' + curprov[2]]
                    fout_str.write('\t'.join(curstr) + '\n')
                    items[1] = ':String_' + str(str_count)
                    str_count += 1
                    output = [items[0], rel, items[1], document[0] + ':' + curprov[2] + ';' + curprov[0],
                              str(probs[count][ele - 1])]
                else:
                    output = [items[0], rel, items[1], curprov[0], str(probs[count][ele - 1])]
                fout.write('\t'.join(output) + '\n')
                if rel == 'org:top_members_employees':
                    output = [items[1], 'per:employee_or_member_of', items[0], curprov[0], str(probs[count][ele - 1])]
                    fout.write('\t'.join(output) + '\n')
                elif (rel == 'per:origin' or rel == 'per:employee_or_member_of') and items[3] == 'COUNTRY':
                    output = [items[0], 'per:countries_of_residence', items[1], curprov[0], str(probs[count][ele - 1])]
                    fout.write('\t'.join(output) + '\n')
                elif rel == 'org:top_members_employees' and items[3] == 'COUNTRY':
                    output = [items[1], 'per:countries_of_residence', items[0], curprov[0], str(probs[count][ele - 1])]
                    fout.write('\t'.join(output) + '\n')
    count += 1
    if count % 200 == 0:
        print count

fin1.close()
fin2.close()
fout.close()
fout_str.close()
