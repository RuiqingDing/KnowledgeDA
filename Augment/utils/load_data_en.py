import time
import pandas as pd
from copy import deepcopy
from ordered_set import OrderedSet

def read_file(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    content = f.read()
    return eval(content)

def load_data(kg_file):
    '''
    read kg data
    Return:
    id2ent: {id: entity name}
    ent2id: {entity name: id}
    id2cat: {id : category name}
    cat2id: {category name: id}
    cid2eidlist: {category id: [entity id list]}
    eid2cid: {entity id: category id}
    triples_all: dataframe, columns : [eid1, rid, eid2]
    '''
    ent2cat = read_file(f'{kg_file}/TagKG/dict_cui.txt')
    category2entlist = {}
    for ent in ent2cat:
        cat = ent2cat[ent]
        if cat not in category2entlist:
            category2entlist[cat] = [ent]
        else:
            category2entlist[cat].append(ent) 

    cat_set, ent_set, rel_set = OrderedSet(), OrderedSet(), OrderedSet()
    for ent in ent2cat:
        ent_set.add(ent)
        cat_set.add(ent2cat[ent])
    
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    cat2id = {cat: idx for idx, cat in enumerate(cat_set)}

    id2ent = {idx: ent for ent, idx in ent2id.items()}
    id2cat = {idx: cat for cat, idx in cat2id.items()}

    cid2eidlist, eid2cid = {}, {}
    for cid in id2cat:
        cat = id2cat[cid]
        eidlist = [ent2id[ent] for ent in category2entlist[cat]]
        cid2eidlist[cid] = eidlist
        for eid in eidlist:
            eid2cid[eid] = cid
    triples = pd.read_csv(f'{kg_file}/TagKG/triples.csv', encoding='utf-8')

    rel_set = OrderedSet(triples['rel'].unique())
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    # rel2id.update({rel+'_reverse': idx+len(rel2id) for idx, rel in enumerate(rel_set)})
    id2rel = {idx: rel for rel, idx in rel2id.items()}

    df_tag = pd.DataFrame.from_dict(id2ent, orient='index', columns=['tag_name'])
    df_tag = df_tag.reset_index().rename(columns={'index': 'eid'})
    df_rel = pd.DataFrame.from_dict(id2rel, orient='index', columns=['rel_name'])
    df_rel = df_rel.reset_index().rename(columns={'index': 'rid'})


    triples = triples.merge(df_tag, left_on = 'tag1', right_on = 'tag_name', how='left')
    triples.drop(columns=['tag_name'], inplace=True)
    triples.rename(columns={'eid': 'eid1'}, inplace=True)
    triples = triples.merge(df_tag, left_on = 'tag2', right_on ='tag_name', how='left')
    triples.drop(columns=['tag_name'], inplace=True)
    triples.rename(columns={'eid': 'eid2'}, inplace=True)
    triples = triples.merge(df_rel, left_on = 'rel', right_on = 'rel_name', how='left')
    triples.drop(columns=['rel_name'], inplace=True)
    
    triples = triples[['eid1', 'rid', 'eid2']]
    print(len(triples),'\n', triples.head(5))
    triples.dropna(axis=0, how='any', inplace=True)

    triples_reverse = deepcopy(triples)
    triples_reverse.rename({'eid1': 'eid2', 'eid2': 'eid1'}, inplace=True)
    triples_reverse['rid'] =  triples_reverse['rid'] + len(rel2id)

    triples_all = pd.concat([triples, triples_reverse], ignore_index=True)
    return id2ent, ent2id, id2cat, cat2id, cid2eidlist, eid2cid, triples_all
