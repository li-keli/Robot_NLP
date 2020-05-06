"""语义处理

负责接收外部语义，并指派处理器进行处理
"""
from app import chatBot
from core.handle import get_tuLing_response


def semantic_factory(input_dict):
    """
    语义处理工厂
    """
    reply_msg = str(chatBot.get_response(input_dict, conversation_id=input_dict['token']))
    if not reply_msg:
        reply_msg = get_tuLing_response(input_dict['msg'])
    return reply_msg