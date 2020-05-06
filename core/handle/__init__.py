import logging
import requests

import jieba
import numpy as np
import yaml

from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras.models import model_from_yaml
from keras_preprocessing import sequence

from core import IntentionType

max_len = 100
lstm_set_src = 'config/lstm.yml'
lstm_model_src = 'config/lstm.h5'
word2vec_model_src = 'config/word2vec_model.pkl'

# 加载LSTM网络模型
with open(lstm_set_src, 'r') as f:
    yaml_string = yaml.load(f)
model = model_from_yaml(yaml_string)
model.load_weights(lstm_model_src)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 加载词向量模型
words_model = Word2Vec.load(word2vec_model_src)

logging.info('lstm model and word2vec model load succesfully.')


def create_dictionaries(model=None, combined=None):
    """
    返回索引，单词向量矩阵和具有统一长度和索引的句子
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
        logging.warning('No data provided...')


def predict(statement):
    """
    预测意图
    """
    words = list(jieba.cut(statement))
    words = np.array(words).reshape(1, -1)
    _, _, data = create_dictionaries(words_model, words)
    data.reshape(1, -1)

    # 预测数据
    logging.debug(model.predict_proba(data)[0].tolist())

    # 预测数据
    result = model.predict_classes(data)
    intention = IntentionType(result)
    logging.debug('意图预测结果：%s', intention)

    return intention


def get_tuLing_response(msg):
    """
    图灵（外围服务）
    """
    apiUrl = 'http://openapi.tuling123.com/openapi/api/v2'
    data = {
        'userInfo': {
            "apiKey": "bdd1ac0e36ad44189e82794430ccdf3b",
            "userId": "wechat-robot123"
        },
        "reqType": 0,
        "perception": {
            "inputText": {
                "text": msg
            },
        },
    }
    try:
        reply_json = requests.post(apiUrl, data=data).json()
        print(reply_json)
        reply_msg = reply_json.get('results')
        for msg in reply_msg:
            if msg.get('resultType') == "text":
                return msg.get('resultType').get('values').get('text')
        # if reply_msg != msg and not msg.startswith(reply_msg):
        #     if '图灵' not in reply_msg:
        #         return reply_msg
        return ''
    except:
        return ''
