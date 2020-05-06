"""
多轮对话管理，仅处理一个意图的对话

李科笠 2018年10月25日
"""
from app import r_cache


def get_cache(token, key):
    return r_cache.hget(token, key)


def get_all_cache(token):
    return r_cache.hgetall(token)


def exist_cache(token, key):
    return r_cache.hexists(token, key)


def set_cache(token, key, value, cache_time=120):
    # 处于业务需求，设置为空的key，直接从redis移除，默认2min缓存
    if value is not None and value != '':
        r_cache.hset(token, key, value)
        r_cache.expire(token, cache_time)
    else:
        clear_cache(token, key)


def clear_cache(token, key):
    r_cache.hdel(token, key)


def clear_h(token):
    r_cache.delete(token)
