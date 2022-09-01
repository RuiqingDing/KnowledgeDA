from sklearn.feature_extraction.text import CountVectorizer #文本向量化
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from nltk import word_tokenize

def read_file(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    return eval(content)

def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):  
        return True
    else:
        return False

def get_cluster(filename, output_file, mwetokenizer, n_clusters = 10):
    df = pd.read_csv(f'{filename}', delimiter='\t', header=None)
    df.columns = ['ID', 'label', 'text']

    # 分词
    words=[]
    for i,row in df.iterrows():
        word=mwetokenizer.tokenize(word_tokenize(row['text']))
        result=' '.join(word)
        words.append(result)
    print(f'len(words) = {len(words)}')

    # 文本向量化：建立词频矩阵
    vect=CountVectorizer()
    x = vect.fit_transform(words)
    x = x.toarray()

    # 构造特征矩阵
    words_name = vect.get_feature_names()
    df_fea = pd.DataFrame(x,columns=words_name)
    print(df_fea.shape)
    df_fea_cs = cosine_similarity(df_fea)

    # 模型搭建
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

