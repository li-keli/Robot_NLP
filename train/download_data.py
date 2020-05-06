import csv
import os
import pandas

from pymongo import MongoClient


# 下载数据，若是数据文件存在，则跳过，若是不存在则从DB抽取
def download():
    if not os.path.exists("data/original_data.csv"):
        # 从MongoDB获取最新的聊天语料
        conn = MongoClient('127.0.0.1', 27017)
        db = conn.customer_service_db
        message_collection = db.message
        all_customer_msgs = message_collection.find({"oper_code": 2002}) 
        source_data = []
        for singel_document in all_customer_msgs:
            msg = singel_document["msg"]
            if not msg:
                continue
            source_data.append(str(msg).replace('\n', '').replace(' ', ''))
        source_data = list(set(source_data))

        with open("data/original_data.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, lineterminator='\n')
            for val in source_data:
                wr.writerow([val])


def handle_excel():
    """
    单独处理下载的Excel数据
    :return:
    """
    datas = pandas.read_excel("data/客户语料.xlsx", header=None, sheet_name=None)
    print(datas.keys())
    for k in datas.keys():
        d = datas[k].dropna()
        d[0].to_csv("data/%s.csv" % k, encoding='utf-8', header=None, index=False)


if __name__ == '__main__':
    # download()
    # print("下载完成...")
    handle_excel()
