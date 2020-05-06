"""
【实现原理】
制作训练数据集：将3000左右的对话中，手工分类出问句类别如./train/data_collated/*.csv样例
训练词向量：通过word2vec来训练自己的词向量集
神经网络模型：使用LSTM实现的循环神经网络来训练上一步转换好的词向量集（使用tensorflow引擎）


【语料库】
训练用语料库位置：./train/data_collated/*.csv，训练语料库内容如下：
咨询休息厅：511
其他问题：2861
总量：3372


【训练报告】
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 20, 300)           119100
_________________________________________________________________
lstm_1 (LSTM)                (None, 50)                70200
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0
=================================================================
Total params: 189,351
Trainable params: 189,351
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
2697/2697 [==============================] - 3s 1ms/step - loss: 0.4429 - acc: 0.8068
Epoch 2/10
2697/2697 [==============================] - 2s 797us/step - loss: 0.2884 - acc: 0.8951
Epoch 3/10
2697/2697 [==============================] - 2s 800us/step - loss: 0.2290 - acc: 0.9344
Epoch 4/10
2697/2697 [==============================] - 2s 806us/step - loss: 0.1695 - acc: 0.9633
Epoch 5/10
2697/2697 [==============================] - 2s 792us/step - loss: 0.1316 - acc: 0.9726
Epoch 6/10
2697/2697 [==============================] - 2s 799us/step - loss: 0.1080 - acc: 0.9796
Epoch 7/10
2697/2697 [==============================] - 2s 801us/step - loss: 0.1024 - acc: 0.9844
Epoch 8/10
2697/2697 [==============================] - 2s 790us/step - loss: 0.0846 - acc: 0.9859
Epoch 9/10
2697/2697 [==============================] - 2s 803us/step - loss: 0.0800 - acc: 0.9863
Epoch 10/10
2697/2697 [==============================] - 2s 797us/step - loss: 0.0770 - acc: 0.9859
675/675 [==============================] - 0s 481us/step
Test score: [0.13758430318734435, 0.9733333333333334]


【最终预测结果】
0.9733


【总结】
最终训练结果已经满足绝大部分识别需求，每种分类配上一套槽组进行多轮对话就可以完成任何问句的精准识别了。

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
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.preprocessing import sequence
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
n_epoch = 10
# 输入的形状
input_length = 100
# 逻辑处理器个数
cpu_count = multiprocessing.cpu_count()
# 加载字典
jieba.load_userdict("../config/dict")


def load_file():
    neg = pd.read_csv('data_collated/hall_data.csv', header=None)
    pos = pd.read_csv('data_collated/other_data.csv', header=None)
    neg = np.array(neg[0])
    post = np.array(pos[0])
    return neg, post


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
    if os.path.isfile("data/doc2_original_data.csv"):
        print("处理停用词表")
        stoplist = set([line.strip() for line in open("data/stopword", 'r', encoding='utf-8').readlines()])

    # 将句子用结巴分开，去掉多余的单词，返回列表
    text_list = []
    for document in text:

        seg_list = jieba.cut(document.strip())
        fenci = []

        for item in seg_list:
            if item not in stoplist and re.match(r'-?\d+\.?\d*', item) is None and len(item.strip()) > 0:
                fenci.append(item)
        # 如果句子的分词为空，则相应删除句子的标签
        if len(fenci) > 0:
            text_list.append(fenci)
    return text_list


def tokenizer(neg, post):
    neg_sege = word_sege(neg)
    post_sege = word_sege(post)
    combined = np.concatenate((post_sege, neg_sege))
    # 生成标签和计量标签数据
    y = np.concatenate((np.ones(len(post_sege), dtype=int), np.zeros(len(neg_sege), dtype=int)))
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
    print(x_train.shape, y_train.shape)

    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    """
    定义神经网络结构
    :param n_symbols:
    :param embedding_weights:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    # model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(LSTM(units=50, activation="sigmoid", recurrent_activation="hard_sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
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
    print('加载数据...')
    neg, post = load_file()

    print('分词，打标签...')
    combined, y = tokenizer(neg, post)
    # print(len(combined), len(y))

    print('训练word2vec模型...')
    index_dict, word_vectors, combined = word2vec_train(combined)

    print('转换成标准的输入层数据')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)

    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def lstm_predict(input_str):
    print('加载模型配置')
    with open('data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    model.load_weights('data/lstm.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 加载词向量模型
    words = list(jieba.cut(input_str))
    words = np.array(words).reshape(1, -1)
    words_model = Word2Vec.load('data/word2vec_model.pkl')
    _, _, data = create_dictionaries(words_model, words)
    data = data.reshape(1, -1)

    # 预测数据
    result = model.predict_classes(data)
    print(words, data, result[0][0])

    if result[0][0] == 1:
        print(input_str, ' 咨询其他')
    else:
        print(input_str, ' 咨询休息厅')


if __name__ == '__main__':
    train()
    while 1:
        input_str = input('请输入问题：')
        lstm_predict(input_str)
