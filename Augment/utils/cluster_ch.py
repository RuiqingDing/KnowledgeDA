from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import jieba

jieba.load_userdict('KG/CMedicalKG/entities_list.txt') #add medical entities when cutting sentences

def get_cluster(filename, output_file, n_clusters = 10):
    df = pd.read_csv(f'{filename}', delimiter='\t', header=None)
    df.columns = ['ID', 'label', 'text']

    words=[]
    for i,row in df.iterrows():
        word=jieba.cut(row['text'])
        result=' '.join(word)
        words.append(result)
    print(f'len(words) = {len(words)}')

    vect=CountVectorizer()
    x = vect.fit_transform(words)
    x = x.toarray()

    words_name = vect.get_feature_names()
    df_fea = pd.DataFrame(x,columns=words_name)
    print(df_fea.shape)
    df_fea_cs = cosine_similarity(df_fea)

    
    kms = KMeans(n_clusters = n_clusters, random_state = 0)
    label_kms = kms.fit_predict(df_fea_cs)
    df['cluster'] = label_kms

    print(df.head(5))

    dict_cluster = {}
    for index, row in df.iterrows():
        cluster = int(row['cluster'])
        qid  =row['ID']
        dict_cluster[qid] = cluster

    f = open(f'{output_file}/dict_cluster_kmeans.txt', 'w', encoding='utf-8')
    f.write(str(dict_cluster))
    f.close()
