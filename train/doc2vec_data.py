import os
import csv
import jieba.analyse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pymongo import MongoClient

jieba.load_userdict("../config/dict")
# 停用词列表
stopwords = [line.strip() for line in open("../config/stopword", 'r', encoding='utf-8').readlines()]


def load_data():
    """
    加载数据，若是数据文件存在，则跳过，若是不存在则从DB抽取

    :return:
    """
    if not os.path.exists("data/doc2_original_data.csv"):
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
        with open("data/doc2_original_data.csv", "w", newline="", encoding='utf-8') as f:
            wr = csv.writer(f, lineterminator='\n')
            for val in source_data:
                wr.writerow([val])

        # 分词处理
        with open("data/doc2_original_data_cut.csv", "w", newline="", encoding='utf-8') as f:
            wr = csv.writer(f, lineterminator='\n')
            for val in list(set(source_data)):
                # 停用词过滤
                # sentence = [t.word.strip() for t in pseg.cut(val) if t not in stopwords and t.flag not in ['w', 'x']]
                sentence = jieba.analyse.extract_tags(val)
                if sentence:
                    wr.writerow([' '.join(sentence)])


load_data()

x_train = [TaggedDocument(doc, [i]) for i, doc in enumerate([s.split(" ") for s in open("data/doc2_original_data_cut.csv", 'r', encoding='utf-8').readlines()])]

model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=100, sample=1e-3, negative=5, workers=4)
model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=90)  # model_dm.iter
model_dm.save('data/doc2_original_data_vector.pkl')

# test
test_text = list(jieba.cut('南宁有你们的厅吗？'))
inferred_vector_dm = model_dm.infer_vector(test_text)
sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
for count, sim in sims:
    sentence = x_train[count]
    words = ''
    for word in sentence[0]:
        words += word
    print('步长: %f %s' % (sim, words))
