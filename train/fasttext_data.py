"""
围绕FaceBook的FastText实现。
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb

FastText和Word2Vec的具体性能测试结果和结论：
https://rare-technologies.com/fasttext-and-gensim-word-embeddings/#conclusions

经过测试，效果没有想象中那么理想，但是用起来确实简单，中文只需要分词后丢进去就行了。

pip install Cython fasttext

李科笠 2018年09月29日
"""

import os
import csv
import jieba.analyse
from pymongo import MongoClient
import fasttext

targ_hall = ['服务中心', '贵宾室', '服务区', '服务厅', '休息室', '休息厅', '驿站']

if not os.path.exists("data/fast_original_data.csv"):
    # 从MongoDB获取最新的聊天语料
    conn = MongoClient('127.0.0.1', 27017)
    db = conn.customer_service_db
    message_collection = db.message
    all_customer_msgs = message_collection.find({"oper_code": 2002})  # {"oper_code": 2002}
    source_data = []
    for singel_document in all_customer_msgs:
        msg = singel_document["msg"]
        if not msg:
            continue
        source_data.append(str(msg).replace('\n', '').replace(' ', ''))

    # 存储源数据
    with open("data/fast_original_data.csv", "w", newline="", encoding='utf-8') as f:
        wr = csv.writer(f, lineterminator='\n')
        for val in source_data:
            source_texts = list(jieba.cut(val))
            # TODO 粗糙打标签，正式使用应该手工分标签
            if len(set(source_texts).intersection(targ_hall)) > 0:
                wr.writerow(['__lable__1 ' + ' '.join(source_texts)])
            else:
                wr.writerow(['__lable__0 ' + ' '.join(source_texts)])

classifier = fasttext.supervised('data/fast_original_data.csv', 'data/fast_original_data_model', epoch=20, label_prefix='__lable__')
cut_text = [x for x in jieba.cut("天津休息厅电话多少")]

print(cut_text)
print(classifier.predict(cut_text))
print(classifier.predict_proba(cut_text))
