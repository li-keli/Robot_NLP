"""
神经网络分类器已经准备完毕，由于可实践数据量过少（<2000），无法做有效的模型训练
但经过测试，通过自己的预料自行训练的词向量，然后组合向量矩阵（句子）可以进行高精度多分类。

预计预料素材需要达到10w+左右，可以进行有效的分类计算。
"""
import csv

import jieba.posseg as pseg
import jieba.analyse
import numpy as np
from gensim.models import Word2Vec
from pymongo import MongoClient

# 停用词列表
stopwords = [line.strip() for line in open("../config/stopword", 'r', encoding='utf-8').readlines()]


# 从MongoDB获取最新的聊天语料
def get_data():
    conn = MongoClient('127.0.0.1', 27017)
    db = conn.customer_service_db
    message_collection = db.message

    all_customer_msgs = message_collection.find() # {"oper_code": 2002}

    for singel_document in all_customer_msgs:
        msg = singel_document["msg"]
        if not msg:
            continue
        yield str(msg).replace('\n', '').replace(' ', '')


# 存储源数据
with open("data/word2_original_data.csv", "w", newline="", encoding='utf-8') as f:
    wr = csv.writer(f, lineterminator='\n')
    for val in get_data():
        wr.writerow([val])

jieba.load_userdict("../config/dict")
# 分词处理
with open("data/word2_original_data_cut.csv", "w", newline="", encoding='utf-8') as f:
    wr = csv.writer(f, lineterminator='\n')
    for val in get_data():
        # 停用词过滤
        sentence = [t.word.strip() for t in pseg.cut(val) if t not in stopwords and t.flag not in ['w', 'x']]
        if sentence:
            wr.writerow([' '.join(sentence)])


def buildWordVector(text, size, imdb_w2v):
    """
    对每个句子的所有词向量取均值, 0补位
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


x_train = [t.split(" ") for t in open("data/word2_original_data_cut.csv", 'r', encoding='utf-8').readlines()]
n_dim = 300
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)
imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
train_vecs = np.concatenate([buildWordVector(z, n_dim, imdb_w2v) for z in x_train])
np.save('data/word2_original_data_vector.npy', train_vecs)

for vocad in imdb_w2v.wv.most_similar(['休息厅']):
    print(vocad)
