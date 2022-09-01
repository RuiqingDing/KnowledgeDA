from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import time
import jieba
import os

jieba.load_userdict(f'../KG/CMedicalKG/entities_list.txt')

def replace_entity(text, ent2id, eid2cid, id2cat):
    '''
    replace the specific entity name with [category]
    '''    
    seg_list = jieba.cut(text)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    for word in words:
        if word in ent2id:
            eid = ent2id[word]
            cid = eid2cid[eid]
            category = id2cat[cid]
            text = text.replace(word, '['+category+']')
    return text


def generate_abstract_data(data_file, input_file, output_file, ent2id, eid2cid, id2cat):
    start = time.time()
    f  = open(f'{input_file}', 'r', encoding = 'utf-8')
    if not os.path.exists(f'{data_file}'):
        os.mkdir(f'{data_file}')
        print(f'Created file: {data_file}')

    f_w = open(f'{output_file}', 'w', encoding = 'utf-8')
    print(f'\nStart to abstract: {output_file} --------- ')

    cnt = 0
    for line in f.readlines():
        cnt += 1
        if cnt % 1000 == 0:
            print(f'No. {cnt}: {time.time() - start}')
        cc = line.split('\t')
        qid = cc[0]
        label = cc[1]
        text = cc[2]
        text_re = replace_entity(text, ent2id, eid2cid, id2cat)
        f_w.write(qid + '\t' + label + '\t' + text_re)

    f.close()
    f_w.close()
    print(f'Finish {output_file}, spend {time.time() - start}')
    
