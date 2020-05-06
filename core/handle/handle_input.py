import logging
import jieba.posseg as pseg

from chatterbot.conversation import Statement
from chatterbot.input import InputAdapter

from core import IntentionType
from core.handle import predict
from core.help.cache import get_cache, set_cache


# 小金的输入适配器
class JinInputAdapter(InputAdapter):

    def process_input(self, input_dict):
        st = Statement(input_dict['msg'])

        # 令牌
        st.add_extra_data("token", input_dict['token'])

        # 分词
        st.add_extra_data("statement_cut", [(t_word.word, t_word.flag) for t_word in pseg.lcut(input_dict['msg'])])
        logging.debug(st.extra_data['statement_cut'])

        # 意图
        log_intention = get_cache(input_dict['token'], 'intention')
        ai_intention = predict(input_dict['msg'])
        intention = ai_intention if log_intention is None else IntentionType(int(log_intention))

        st.add_extra_data("statement_intention", intention)
        logging.debug('log_intention is: %s, ai_intention is: %s, the end intention is: %s', log_intention, ai_intention, st.extra_data['statement_intention'])

        # 记忆话题
        if intention != IntentionType.Other:
            set_cache(input_dict['token'], 'intention', intention.value)

        return st
