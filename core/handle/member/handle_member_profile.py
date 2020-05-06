"""会员卡

处理会员基础信息问题
"""
import logging

from chatterbot.logic import LogicAdapter

from core import IntentionType
from core.help.cache import get_all_cache, set_cache, clear_h


class MemberProfileAdapter(LogicAdapter):

    def __init__(self, **kwargs):
        super(MemberProfileAdapter, self).__init__(**kwargs)

    def can_process(self, statement):
        return statement.extra_data.get('statement_intention', IntentionType.Other) == IntentionType.VipCard

    def process(self, statement):
        import nltk
        from core.help.phone import extract_phone_numbers
        from depend.server_member_info import SearchMemberInfo

        extra_data = statement.extra_data
        grams = "phone:{<m>}"
        parser = nltk.RegexpParser(grams)
        parsed_tree = parser.parse(extra_data['statement_cut'])

        # 抽取实体
        phone = ''
        for tree in parsed_tree.subtrees(lambda x: x.height() == 2):
            tree_label = tree.label()
            # 手机号抽取
            if tree_label == 'phone':
                phone = extract_phone_numbers([x for x in tree.leaves()][0][0])
                set_cache(extra_data['token'], 'phone', phone)
        logging.debug('抽取出的实体：%s', phone)

        is_pass, miss_msg = self._check_notch(extra_data)
        if not is_pass:
            return self.__return_response(miss_msg)

        dm = get_all_cache(extra_data['token'])
        reply_msg = self.__return_response(SearchMemberInfo().get_cust_profile_describe(dm['phone']))

        clear_h(extra_data['token'])
        return reply_msg

    # 槽口检查
    def _check_notch(self, extra_data):
        notchs = get_all_cache(extra_data['token'])
        logging.info('槽口检查: %s', notchs)
        if notchs:
            if 'phone' not in notchs.keys():
                return False, "请告诉我您的注册手机号？"
        return True, ""

    def __return_response(self, msg, confidence=1):
        """
        返回Statement结构
        :param msg:
        :param confidence:
        :return:
        """
        from chatterbot.conversation import Statement

        st = Statement(msg)
        st.confidence = confidence
        return st
