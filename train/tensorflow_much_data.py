"""
tensorflow 多分类

李科笠 2018年10月15日
"""
import multiprocessing
import re
import os

import jieba
import numpy as np
import pandas as pd
import yaml
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
from keras.applications.resnet50 import decode_predictions
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 随机数种子
np.random.seed(1337)
# 词向量的维数
vocab_dim = 300
# 最大句子长度
max_len = 100
# 在语料库上的迭代次数(epoch)
n_iterations = 1
# 忽略总频率低于这个值的所有单词
n_exposures = 10
# 一个句子中当前单词和预测单词之间的最大距离
window_size = 7
# 每个梯度更新的样本数量
batch_size = 32
# 迭代次数
n_epoch = 20
# 输入的形状
input_length = 100
# 逻辑处理器个数
cpu_count = multiprocessing.cpu_count()
# 加载字典
jieba.load_userdict("../config/dict")


def get_stop_word(stopword_path):
    stoplist = set()
    for line in stopword_path:
        stoplist.add(line.strip())
    return stoplist


def word_sege(text):
    """
    分词去除停用词
    """
    stoplist = set()
    if os.path.isfile("data/stopword"):
        print("处理停用词表")
        stoplist = set([line.strip() for line in open("data/stopword", 'r', encoding='utf-8').readlines()])

    # 将句子用结巴分开，去掉多余的单词，返回列表
    text_list = []
    for document in text:
        seg_list = jieba.cut(str(document).strip())
        fenci = []

        for item in seg_list:
            if item not in stoplist and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 0:
                fenci.append(item)
        # 如果句子的分词为空，则相应删除句子的标签
        if len(fenci) > 0:
            text_list.append(fenci)
    return text_list


def tokenizer():
    data_frames = pd.read_excel('data/客户语料.xlsx', header=None, sheet_name=None)

    combined, y, i = [], np.empty(shape=(0, 0), dtype=int), 0
    for k in data_frames.keys():
        sege = word_sege(np.array(data_frames[k][0]))
        combined = np.append(combined, sege)
        y = np.append(y, np.full(len(sege), i, dtype=int))
        i += 1
    return combined, y


def create_dictionaries(model=None, combined=None):
    """
    返回索引，单词向量矩阵和具有统一长度和索引的句子

    函数做的是工作的数量:
        1- 创建一个单词来索引映射
        2- 创建一个字到向量映射
        3- 转换培训和测试字典
    """
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        # 有单词向量的单词的索引不为0
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        # 将所有对应的向量整合到向量矩阵中
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # 用keras的pad_sequences函数统一句子的长度
        combined = sequence.pad_sequences(combined, maxlen=max_len)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def word2vec_train(combined):
    """
    单词向量的训练
    """
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    # 建立词汇字典
    model.build_vocab(combined)
    # 训练单词向量模型
    model.train(combined, total_examples=model.corpus_count, epochs=50)
    model.save('data/word2vec_model.pkl')
    # 索引、词向量矩阵和基于训练模型的统一长度和索引的句子
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    """
    返回lstm模型所需的输入参数
    """
    n_symbols = len(index_dict) + 1
    # 构建对应于索引的字向量矩阵
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    # 划分测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    # one-hot结构
    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)
    # print(x_train.shape, y_train.shape)

    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    """
    定义神经网络结构
    """
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(units=50, activation="tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))  # sigmoid

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1)

    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    # 保存配置模型
    yaml_string = model.to_yaml()
    with open('data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('data/lstm.h5')
    print('Test score:', score)


def train():
    print('加载数据，分词，打标签...')
    combined, y = tokenizer()
    print('数据规模', len(combined), len(y), y)

    print('train word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)

    print('输入层数据...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('data/word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(input_str):
    # print('加载模型配置')
    with open('data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    model.load_weights('data/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 加载词向量模型
    data = input_transform(input_str)
    data.reshape(1, -1)

    # 预测数据
    result = model.predict_proba(data)[0]
    # result = model.predict_classes(data)
    np.set_printoptions(precision=4, suppress=True)

    print(result)


if __name__ == '__main__':
    train()
    while 1:
        input_str = input('请输入问题：')
        # input_str = '上海休息厅电话是多少'
        lstm_predict(input_str)
