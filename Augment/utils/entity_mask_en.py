from copy import deepcopy
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import time
import os

def replace_entity(text, mwetokenizer, stop_words, ent2id, eid2cid):
    '''
    replace the specific entity name with [category]
    '''    
    seg_list = mwetokenizer.tokenize(word_tokenize(text))
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    for word in words:
        if word in ent2id and word not in stop_words:
            eid = ent2id[word]
            cid = eid2cid[eid]
            # print(word, ', ', id2cat[cid])
            # category = id2cat[cid]
            text = text.replace(word, '['+str(int(cid))+']')
    return text


def generate_abstract_data(data_file, input_file, output_file, mwetokenizer, stop_words, ent2id, eid2cid):
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
        text_re = replace_entity(text, mwetokenizer, stop_words, ent2id, eid2cid)
        f_w.write(qid + '\t' + label + '\t' + text_re)

    f.close()
    f_w.close()
    print(f'Finish {output_file}, spend {time.time() - start}')

