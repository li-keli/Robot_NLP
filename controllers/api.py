# coding=utf-8
import logging

from app import app
from flask import request
from core.manager import semantic_factory


@app.route('/health')
def health():
    """
    健康检查
    :return:
    """
    return 'success\n'


@app.route('/semantic', methods=["GET", "POST"])
def semantic():
    """
    对话处理

    :return: answer
    """
    try:
        json_data = request.get_json()
        if not json_data['msg']:
            return ''
        
        return semantic_factory(json_data)
    except AssertionError as ex:
        logging.exception(ex)

        return str(ex)
    except Exception as ex:
        logging.exception(ex)

        return ''
