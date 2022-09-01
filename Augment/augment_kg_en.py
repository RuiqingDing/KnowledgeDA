#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
import time
import itertools
import warnings
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
from utils.load_data_en import load_data
from utils.entity_mask_en import generate_abstract_data
from utils.cluster_en import get_cluster
from utils.file_handler import read_file, write_file
warnings.filterwarnings("ignore")


# load data
start = time.time()
id2ent, ent2id, id2cat, cat2id, cid2eidlist, eid2cid, triples_all = load_data('../KG')
print(f'Finish loading data, spend {time.time() - start}')

def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):  
        return True
    else:
        return False


# preprocess TagKG, delete stopwords and non-English strings
start = time.time()
stop_words = set(stopwords.words('english'))
entities = []
del_eids, del_ents = [], []
for ent in ent2id:
    if not is_alphabet(ent):
        eid = ent2id[ent]
        del_eids.append(eid)
        del_ents.append(ent)
    elif ent in stop_words:
        eid = ent2id[ent]
        del_eids.append(eid)
        del_ents.append(ent)
    else:
        word_tuple = tuple(ent.split(' '))
        entities.append(word_tuple)

# delete stopwords
for eid in del_eids:
    id2ent.pop(eid)
    eid2cid.pop(eid)

for ent in del_ents:
    ent2id.pop(ent)

for cid in cid2eidlist:
    eidlist = cid2eidlist[cid]
    eidlist = [eid for eid in eidlist if eid not in del_eids]
    cid2eidlist[cid] = eidlist

print(f'---Origin triples number: {len(triples_all)}')
triples_all = triples_all[~triples_all.eid1.isin(del_eids)]
triples_all = triples_all[~triples_all.eid2.isin(del_eids)]
print(f'---delete entity number: {len(del_eids)}')
print(f'---triples number after deleting: {len(triples_all)}')
print(f'spend {time.time()-start}\n')


# # load entity names into dictionary
entities = read_file(f'../KG/TagKG/entities_list.txt') #这个是umls的医疗数据集的子集
print('Loading tagKG...')
print('entity number: ', len(entities))
mwetokenizer = MWETokenizer(entities, separator=' ')


def synonym_replace(words, n):
    # from EDA
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break
	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')
	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)


def get_entity_sim(eid, rids=[], n=1, must_n=True):
    cid = eid2cid[eid]
    # find entities with same category
    eids_with_samecid = set(cid2eidlist[cid])-set([eid])
    eids_candidate = list(eids_with_samecid)

    if len(rids) > 0:
        eids_with_rids = set(triples_all[triples_all.rid.isin(rids)]['eid1'].unique())
        eids_candidate = list(set(eids_candidate).intersection(eids_with_rids))

    if must_n:
        if len(eids_candidate) >= n:
            eid_chosen = [random.choice(eids_candidate)  for i in range(n)]
        else: # there is no entity with same category in 2hop neighbors
            eid_chosen = eids_candidate + [random.choice(list(eids_with_samecid)) for _ in range(n-len(eids_candidate))]
    else:
        if n == 0: # return all candidates
            eid_chosen = eids_candidate
        else:
            n = min(n, len(eids_candidate))
            eid_chosen = [random.choice(eids_candidate)  for i in range(n)]

    if n == 1:
        eid_chosen = eid_chosen[0]
    return eid_chosen

def get_entity_sim2(eid):
    cid = eid2cid[eid]
    eids = set(id2ent.keys())
    eids_with_samecid = (set(cid2eidlist[cid]).intersection(eids))-set([eid])
    eid_chosen = random.choice(list(eids_with_samecid))
    return eid_chosen

def tag_entity_replace(words, entity):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))     
    random.shuffle(random_word_list)

    # eid_chosen = get_entity_sim(entity)
    eid_chosen = get_entity_sim2(entity)
    ent_chosen = id2ent[eid_chosen]

    new_words = [ent_chosen if word  == id2ent[entity] else word for word in new_words]
    return new_words


def tag_triples_replace(words, entities, triples_all=triples_all):
    random_word_list = list(set([word for word in words if word not in stop_words]))     
    random.shuffle(random_word_list)

    eids = entities
    combinations = itertools.combinations(eids, 2)
    coms_keep = {} # which appeared in triples_all 
    for com in combinations:
        if len(triples_all[(triples_all.eid1 == com[0]) & (triples_all.eid2 == com[1])]) > 0:
            rid = triples_all[(triples_all.eid1 == com[0]) & (triples_all.eid2 == com[1])]['rid'].tolist()[0]
            coms_keep[com] = rid
    
    if len(coms_keep) == 0: # eids with no relationship
        for entity in eids:
            new_words = tag_entity_replace(words, entity)
        return new_words

    elif len(coms_keep) == 1:
        for com in coms_keep:
            rid = coms_keep[com]
            eids_sim = get_entity_sim(com[0], n=5)
            triples_rid = triples_all[(triples_all.eid1.isin(eids_sim))&(triples_all.rid == rid)]
            if len(triples_rid) == 0:
                for entity in eids:
                    new_words = tag_entity_replace(words, entity)
                return new_words
            else:
                triples_rid = triples_rid.sample(frac=1.0).reset_index(drop=True)
                eid1 = triples_rid.loc[0, 'eid1']
                eid2 = triples_rid.loc[0, 'eid2']
                ent1_chosen = id2ent[eid1]
                ent2_chosen = id2ent[eid2]
                new_words = words
                new_words = [ent1_chosen if word == id2ent[com[0]] else word for word in new_words]
                new_words = [ent2_chosen if word == id2ent[com[1]] else word for word in new_words]
                return new_words
    else:
        triples = pd.DataFrame(columns=['eid1', 'rid', 'eid2'])
        i = 0
        for com in coms_keep:
            rid = coms_keep[com]
            triples.loc[i, 'eid1'] = com[0]
            triples.loc[i, 'eid2'] = com[1]
            triples.loc[i, 'rid'] = rid
            i += 1
        head_eid_count = triples.groupby('eid1')['rid'].count().reset_index()
        head_eid_count.sort_values(by='rid', ascending=False, inplace=True)
        head_eid_count.reset_index(inplace=True)
        head_eid_top = head_eid_count.loc[0, 'eid1']
        rids = list(triples[triples.eid1 == head_eid_top]['rid'].unique())
        eid_sim = get_entity_sim(head_eid_top, rids, n=1)
        triples.loc[triples.eid1==head_eid_top, 'eid1_replace'] = int(eid_sim)
        triples.loc[triples.eid2==head_eid_top, 'eid2_replace'] = int(eid_sim)
        triples.fillna(9999999, inplace=True)
        triples.sort_values(by=['eid1_replace', 'eid2_replace'], ascending=[True, True], inplace=True)

        # print(triples)
        for _, row in triples.iterrows():
            head = row['eid1_replace']
            tail = row['eid2_replace']
            rel = row['rid']
            if (head == 9999999) and (tail != 9999999):
                head = get_head_entity(tail, rel)
                if head != '':
                    triples.loc[triples.eid1 == row['eid1'], 'eid1_replace'] = head
                    triples.loc[triples.eid2 == row['eid1'], 'eid2_replace'] = head
                else:
                    triples.loc[triples.eid1 == row['eid1'], 'eid1_replace'] = row['eid1']
                    triples.loc[triples.eid2 == row['eid1'], 'eid2_replace'] = row['eid1']
                
            else:
                if (head != 9999999) and (tail == 9999999):
                    tail = get_tail_entity(head, rel)
                    if tail != '':
                        triples.loc[triples.eid1 == row['eid2'], 'eid1_replace'] = tail
                        triples.loc[triples.eid2 == row['eid2'], 'eid2_replace'] = tail
                    else:
                        triples.loc[triples.eid1 == row['eid2'], 'eid1_replace'] = row['eid2']
                        triples.loc[triples.eid2 == row['eid2'], 'eid2_replace'] = row['eid2']

        # print(triples)
        replaces = []
        for _, row in triples.iterrows():
            eid1 = row['eid1']
            eid2 = row['eid2']
            eid1_re = row['eid1_replace']
            eid2_re = row['eid2_replace']
            if eid1 != eid1_re:
                if (eid1_re != 9999999) and ([eid1, eid1_re] not in replaces):
                    replaces.append([eid1, eid1_re])
            if eid2 != eid2_re:
                if (eid2_re != 9999999) and ([eid2, eid2_re] not in replaces):
                    replaces.append([eid2, eid2_re])
        # print(replaces)

        new_words = words
        for replace in replaces:
            eid_origin = replace[0]
            eid_replace = replace[1]
            new_words = [id2ent[eid_replace] if word == id2ent[eid_origin] else word for word in new_words]
        return new_words


def get_tail_entity(head, rel, triples_all = triples_all):
    '''
    find tail entity given head and rel
    '''
    triples_sub = triples_all[(triples_all.eid1 == head) & (triples_all.rid == rel)]
    if len(triples_sub) >= 1:
        triples_sub = triples_sub.sample(frac=1.0).reset_index(drop=True)
        tail = triples_sub.loc[0, 'eid2']
        return int(tail)
    else:
        return ''

def get_head_entity(tail, rel, triples_all = triples_all):
    '''
    find head entity given tail and rel
    '''
    triples_sub = triples_all[(triples_all.eid2 == tail) & (triples_all.rid == rel)]
    if len(triples_sub) >= 1:
        triples_sub = triples_sub.sample(frac=1.0).reset_index(drop=True)
        head = triples_sub.loc[0, 'eid1']
        return int(head)
    else:
        return ''


def tag_kg_replacement(words, ent2id=ent2id):
    random_word_list = list(set([word for word in words if word not in stop_words]))
    entities = [ent2id[word] for word in random_word_list if word in ent2id]
    if len(entities) == 0:
        return synonym_replace(words, min(np.ceil(0.1*len(words)), 15))
    elif len(entities) == 1:
        new_words = tag_entity_replace(words, entities[0])
        return synonym_replace(new_words, min(np.ceil(0.1*len(words)), 15))
    else:
        # return tag_triples_replace(words, entities)
        for entity in entities:
            new_words = tag_entity_replace(words, entity)
        return synonym_replace(new_words, min(np.ceil(0.1*len(words)), 15))


def tag_entity_retrieve(words, entity, id2ent):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))     
    random.shuffle(random_word_list)

    eid = entity
    cid = eid2cid[eid]
    eids = set(id2ent.keys())
    eids_with_samecid = (set(cid2eidlist[cid]).intersection(eids))-set([eid])
    eid_chosen = random.choice(list(eids_with_samecid))
    ent_chosen = id2ent[eid_chosen]
    new_words = [ent_chosen if word  == id2ent[entity] else word for word in new_words]
    return new_words


def tag_triples_retrieve(words, entities, id2ent, triples_all):
    random_word_list = list(set([word for word in words if word not in stop_words]))     
    random.shuffle(random_word_list)

    eids = entities
    combinations = itertools.combinations(eids, 2)
    coms_keep = {} # which appeared in triples_all 
    for com in combinations:
        if len(triples_all[(triples_all.eid1 == com[0]) & (triples_all.eid2 == com[1])]) > 0:
            rid = triples_all[(triples_all.eid1 == com[0]) & (triples_all.eid2 == com[1])]['rid'].tolist()[0]
            coms_keep[com] = rid
    
    if len(coms_keep) == 0: # eids with no relationship
        for entity in eids:
            new_words = tag_entity_retrieve(words, entity, id2ent)
        return new_words
    elif len(coms_keep) == 1:
        for com in coms_keep:
            rid = coms_keep[com]
            triples_rid = triples_all[(triples_all.eid1 != com[0]) & (triples_all.rid == rid)]
            if len(triples_rid) == 0:
                for entity in eids:
                    new_words = tag_entity_retrieve(words, entity, id2ent)
                return new_words
            else:
                triples_rid = triples_rid.sample(frac=1.0).reset_index(drop=True)
                eid1 = triples_rid.loc[0, 'eid1']
                eid2 = triples_rid.loc[0, 'eid2']
                ent1_chosen = id2ent[eid1]
                ent2_chosen = id2ent[eid2]
                new_words = words
                new_words = [ent1_chosen if word == id2ent[com[0]] else word for word in new_words]
                new_words = [ent2_chosen if word == id2ent[com[1]] else word for word in new_words]
                return new_words
    else:
        triples = pd.DataFrame(columns=['eid1', 'rid', 'eid2'])
        i = 0
        for com in coms_keep:
            rid = coms_keep[com]
            triples.loc[i, 'eid1'] = com[0]
            triples.loc[i, 'eid2'] = com[1]
            triples.loc[i, 'rid'] = rid
            i += 1
        head_eid_count = triples.groupby('eid1')['rid'].count().reset_index()
        head_eid_count.sort_values(by='rid', ascending=False, inplace=True)
        head_eid_count.reset_index(inplace=True)
        head_eid_top = head_eid_count.loc[0, 'eid1']
        rids = list(triples[triples.eid1 == head_eid_top]['rid'].unique())
        eids_replace = list(triples_all[triples_all.rid.isin(rids)]['eid1'].unique())
        eid_sim = random.choice(eids_replace)
        
        triples.loc[triples.eid1==head_eid_top, 'eid1_replace'] = int(eid_sim)
        triples.loc[triples.eid2==head_eid_top, 'eid2_replace'] = int(eid_sim)
        triples.fillna(9999999, inplace=True)
        triples.sort_values(by=['eid1_replace', 'eid2_replace'], ascending=[True, True], inplace=True)

        # print(triples)
        for _, row in triples.iterrows():
            head = row['eid1_replace']
            tail = row['eid2_replace']
            rel = row['rid']
            if (head == 9999999) and (tail != 9999999):
                head = get_head_entity(tail, rel, triples_all)
                if head != '':
                    triples.loc[triples.eid1 == row['eid1'], 'eid1_replace'] = head
                    triples.loc[triples.eid2 == row['eid1'], 'eid2_replace'] = head
                else:
                    triples.loc[triples.eid1 == row['eid1'], 'eid1_replace'] = row['eid1']
                    triples.loc[triples.eid2 == row['eid1'], 'eid2_replace'] = row['eid1']
                
            else:
                if (head != 9999999) and (tail == 9999999):
                    tail = get_tail_entity(head, rel, triples_all)
                    if tail != '':
                        triples.loc[triples.eid1 == row['eid2'], 'eid1_replace'] = tail
                        triples.loc[triples.eid2 == row['eid2'], 'eid2_replace'] = tail
                    else:
                        triples.loc[triples.eid1 == row['eid2'], 'eid1_replace'] = row['eid2']
                        triples.loc[triples.eid2 == row['eid2'], 'eid2_replace'] = row['eid2']

        # print(triples)
        replaces = []
        for _, row in triples.iterrows():
            eid1 = row['eid1']
            eid2 = row['eid2']
            eid1_re = row['eid1_replace']
            eid2_re = row['eid2_replace']
            if eid1 != eid1_re:
                if (eid1_re != 9999999) and ([eid1, eid1_re] not in replaces):
                    replaces.append([eid1, eid1_re])
            if eid2 != eid2_re:
                if (eid2_re != 9999999) and ([eid2, eid2_re] not in replaces):
                    replaces.append([eid2, eid2_re])

        new_words = words
        for replace in replaces:
            eid_origin = replace[0]
            eid_replace = replace[1]
            new_words = [id2ent[eid_replace] if word == id2ent[eid_origin] else word for word in new_words]
        return new_words


def tag_retrieve_replacement(words, ent2id, triples_all):
    random_word_list = list(set([word for word in words if word not in stop_words]))
    entities = [ent2id[word] for word in random_word_list if word in ent2id]
    if len(entities) == 0:
        return synonym_replace(words, min(np.ceil(0.1*len(words)), 15))
    elif len(entities) == 1:
        new_words =  tag_entity_retrieve(words, entities[0], id2ent)
        return synonym_replace(new_words, min(np.ceil(0.1*len(words)), 15))
    else:
        # return tag_triples_retrieve(words, entities, id2ent, triples_all)
        for entity in entities:
            new_words = tag_entity_retrieve(words, entity, id2ent)
        return synonym_replace(new_words, min(np.ceil(0.1*len(words)), 15))


def augment_with_kg(sentence):
    seg_list = mwetokenizer.tokenize(word_tokenize(sentence))
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    # print(words)
    a_words = tag_kg_replacement(words)
    return (' '.join(a_words))

    
def generate_augment_data(data_file, input_file, aug_num = 5):
    start = time.time()
    df = pd.read_csv(f'{input_file}', delimiter='\t', header=None)
    df.columns = ['ID', 'label', 'text']
    f = open(f'{input_file}', 'r', encoding='utf-8')

    f_w = open(f'{data_file}/KGER.txt', 'w', encoding = 'utf-8')
    print(f'\nStart to augment: {data_file}/KGER.txt --------- ')

    cnt = 0
    for line in f.readlines():
        cnt += 1
        if cnt % 1000 == 0: 
            print(f'No. {cnt}: {time.time() - start}')
        cc = line.split('\t')
        qid = cc[0]
        label = cc[1]
        text = cc[2]
        f_w.write(qid + '\t' + label + '\t' + text)
        for _ in range(aug_num):
            text_aug = augment_with_kg(text)
            f_w.write(qid + '\t' + label + '\t' + text_aug+'\n')
            # if num == aug_num -1: print(text_aug)
    f.close()
    f_w.close()
    print(f'Finish {data_file}/KGER.txt, spend {time.time() - start}')


def augment_with_retrieve(sentence, ent2id, triples_all):
    seg_list = mwetokenizer.tokenize(word_tokenize(sentence))
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    a_words = tag_retrieve_replacement(words, ent2id, triples_all)
    return (' '.join(a_words))


def generate_cluster_data(data_file, input_file, ent2id, triples_all, qid2cluster, cluster2qidlist, aug_num = 5):
    start = time.time()
    f  = open(f'{input_file}', 'r', encoding = 'utf-8')
    df = pd.read_csv(f'{input_file}', delimiter='\t', header=None)
    df.columns = ['ID', 'label', 'text']

    f_w = open(f'{data_file}/TrainER.txt', 'w', encoding = 'utf-8')
    print(f'\nStart to augment: {data_file}/TrainER.txt --------- ')

    for line in f.readlines():
        cc = line.split('\t')
        qid = cc[0]
        label = cc[1]
        text = cc[2]
        cluster = qid2cluster[qid]
        qids_with_same_label = df[(df.label == label) & (df.ID != qid)]['ID'].tolist()
        qids_select = set(qids_with_same_label) - set(cluster2qidlist[cluster])
        if len(qids_select) > 0:
            triples_sub = triples_all[triples_all.ID.isin(list(qids_select))]
            if len(triples_sub) == 0:
                triples_sub = triples_all
        else:
             triples_sub = triples_all
        for _ in range(aug_num):
            text_aug = augment_with_retrieve(text, ent2id, triples_sub)
            f_w.write(qid + '\t' + label + '\t' + text_aug+'\n')
            
    f.close()
    f_w.close()
    print(f'Finish {data_file}/TrainER.txt, spend {time.time() - start}')



if __name__ == "__main__":
    print('\nGenerate_abstract_data...')
    data_file = f'../data/SO-PLC'
    input_file = data_file + '/train.txt'
    cluster_file = data_file + '/abstract.txt'
    generate_abstract_data(data_file, input_file, cluster_file, mwetokenizer, stop_words, ent2id, eid2cid)

    print('\nKMeans clustering...')
    get_cluster(cluster_file, data_file, mwetokenizer, n_clusters = 10)

    # retrieve from training data
    qid2cluster = read_file(f'{data_file}/dict_cluster_kmeans.txt')
    cluster2qidlist = {}
    for qid in qid2cluster:
        cluster = qid2cluster[qid]
        if cluster not in cluster2qidlist:
            cluster2qidlist[cluster] = [qid]
        else:
            cluster2qidlist[cluster].append(qid)

    print('\nGenerate_TrainER_data...')
    generate_cluster_data(data_file, input_file, ent2id, triples_all, qid2cluster, cluster2qidlist, aug_num = 10)

    print('\nGenerate_KGER_data...')
    generate_augment_data(data_file, input_file, aug_num = 10)

