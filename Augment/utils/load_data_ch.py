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
    category2entlist = read_file(f'{kg_file}/CMedicalKG/entities_dict.txt')

    cat_set, ent_set, rel_set = OrderedSet(), OrderedSet(), OrderedSet()
    for cat in category2entlist:
        cat_set.add(cat)
        ent_set |= OrderedSet(category2entlist[cat])
    
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    cat2id = {cat: idx for idx, cat in enumerate(cat_set)}

    id2ent = {idx: ent for ent, idx in ent2id.items()}
    id2cat = {idx: cat for cat, idx in cat2id.items()}

    cid2eidlist, eid2cid = {}, {}
    for cid in [i for i in range(len(cat2id))]:
        cat = id2cat[cid]
        eidlist = [ent2id[ent] for ent in category2entlist[cat]]
        cid2eidlist[cid] = eidlist
        for eid in eidlist:
            eid2cid[eid] = cid
    triples = pd.read_csv(f'{kg_file}/CMedicalKG/triples.txt', encoding='utf-8', header=None, delimiter='\t')
    triples.columns = ['ent1', 'rel', 'ent2']

    rel_set = OrderedSet(triples['rel'].unique())
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    # rel2id.update({rel+'_reverse': idx+len(rel2id) for idx, rel in enumerate(rel_set)})
    id2rel = {idx: rel for rel, idx in rel2id.items()}

    df_ent = pd.DataFrame.from_dict(id2ent, orient='index', columns=['ent_name'])
    df_ent = df_ent.reset_index().rename(columns={'index': 'eid'})
    df_rel = pd.DataFrame.from_dict(id2rel, orient='index', columns=['rel_name'])
    df_rel = df_rel.reset_index().rename(columns={'index': 'rid'})


    triples = triples.merge(df_ent, left_on = 'ent1', right_on = 'ent_name', how='left')
    triples.drop(columns=['ent_name'], inplace=True)
    triples.rename(columns={'eid': 'eid1'}, inplace=True)
    triples = triples.merge(df_ent, left_on = 'ent2', right_on ='ent_name', how='left')
    triples.drop(columns=['ent_name'], inplace=True)
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

# if __name__ == "__main__":
#     id2ent, ent2id, id2cat, cat2id, cid2eidlist, eid2cid, triples_all = load_data('../KG')