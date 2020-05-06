import logging
import redis
import jieba

from flask import Flask
from chatterbot import ChatBot

# 日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s [ %(filename)s.%(funcName)s.%(lineno)d ] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S %p')

# Redis连接池
redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0, decode_responses=True)
r_cache = redis.StrictRedis(connection_pool=redis_pool)

# 加载词典
jieba.load_userdict('config/dict')

# 机器人引擎配置
chatBot = ChatBot(
    'Jin',
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer',
    preprocessors=['chatterbot.preprocessors.clean_whitespace'],
    input_adapter='core.handle.handle_input.JinInputAdapter',
    logic_adapters=[
        'core.handle.vip_hall.handle_vip_hall.VipHallAdapter',  # 查询休息厅
        'core.handle.member.handle_member_profile.MemberProfileAdapter',  # 查询会员基础信息
        'core.handle.air.handle_air_ticket.AirTicketAdapter',  # 查询机票
        'chatterbot.logic.BestMatch',  # 基础、外围服务
        {
            'import_path': 'chatterbot.logic.LowConfidenceAdapter',
            'threshold': 0.65,
            'default_response': ''
        }
    ],
    storage_adapter='chatterbot.storage.MongoDatabaseAdapter',
    database='customer_service_db',
    database_uri='mongodb://127.0.0.1:27018/',
    read_only=True,
)

# 自定义基础语料
chatBot.train('config/chatter.yml')

# web服务
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 导入API
import controllers
