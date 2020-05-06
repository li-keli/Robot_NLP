"""机票

处理机票相关的咨询问题
"""
from chatterbot.logic import LogicAdapter

from core import IntentionType
from core.help.cache import clear_h


class AirTicketAdapter(LogicAdapter):
    def __init__(self, **kwargs):
        super(AirTicketAdapter, self).__init__(**kwargs)

    def can_process(self, statement):
        extra_data = statement.extra_data
        is_air_ticket = statement.extra_data.get('statement_intention', IntentionType.Other) == IntentionType.AirTicket
        if is_air_ticket:
            clear_h(extra_data['token'])
        return is_air_ticket

    def process(self, statement):
        from chatterbot.conversation import Statement

        st = Statement("")
        st.confidence = 0
        return st
