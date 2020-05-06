"""休息厅

处理休息厅相关的咨询问题
"""
import logging
from chatterbot.logic import LogicAdapter

from core import IntentionType
from core.help.phone import extract_phone_numbers
from core.help.cache import *


class VipHallAdapter(LogicAdapter):
    def __init__(self, **kwargs):
        super(VipHallAdapter, self).__init__(**kwargs)

    def can_process(self, statement):
        return statement.extra_data.get('statement_intention', IntentionType.Other) == IntentionType.HallRoom

    # 咨询休息厅基本信息
    def process(self, statement):
        import nltk
        from depend.server_member_info import SearchMemberInfo
        from depend.server_vip_hall import SearchVipHall

        extra_data = statement.extra_data
        grams = """
                phone:{<m>}
                city:{<ns>}
                airport:
                {<ns><nsf><nsft>}
                {<ns><nsf><nsfg><nsft>}
                {<ns><nsf>}
                {<nsf>}
                """
        parser = nltk.RegexpParser(grams)
        parsed_tree = parser.parse(extra_data['statement_cut'])

        # 抽取实体
        airport, cityname, phone = '', '', ''
        for tree in parsed_tree.subtrees(lambda x: x.height() == 2):
            tree_label = tree.label()
            # 地名实体抽取
            if tree_label == 'city':
                cityname = [x for x in tree.leaves()][0][0]
                set_cache(extra_data['token'], 'city', cityname)
            # 机场抽取
            if tree_label == 'airport':
                airport = [x for x in tree.leaves()][0][0]
                set_cache(extra_data['token'], 'airport', airport)
            # 手机号抽取
            if tree_label == 'phone':
                phone = extract_phone_numbers([x for x in tree.leaves()][0][0])
                set_cache(extra_data['token'], 'phone', phone)
        logging.debug('抽取出的实体：%s %s %s', cityname, airport, phone)

        # 检查槽口
        is_pass, miss_msg = self._check_notch(extra_data, cityname, airport, phone)
        if not is_pass:
            return self.__return_response(miss_msg)

        dm = get_all_cache(extra_data['token'])

        # 检索是否仅有自营厅
        only_proprietary_hall, return_msg = SearchVipHall().proprietary_hall(dm.get('city', ''), dm.get('airport', ''))
        if only_proprietary_hall:
            clear_h(extra_data['token'])  # 一轮对话任务完成，清空数据
            return self.__return_response(return_msg)

        is_pass, miss_msg = self._check_notch(extra_data, cityname, airport, phone, only_proprietary=False)
        if not is_pass:
            return self.__return_response(miss_msg)

        # 查询带有会员信息的休息厅数据
        customer_interests = SearchMemberInfo().get_customer_hall_range(dm['phone'])
        clear_h(extra_data['token'])  # 一轮对话任务完成，清空数据
        if len(customer_interests['VipHallIds']) > 0:
            return self.__return_response(SearchVipHall().vip_hall_with_customer_info(customer_interests, dm.get('city', ''), dm.get('airport', '')))
        else:
            return self.__return_response("您的会员卡在{0}{1}没有查询到可使用的休息厅".format(dm.get('city', ''), dm.get('airport', '')))

    # 槽口检查
    def _check_notch(self, extra_data, cityname, airport, phone, only_proprietary=True):
        # 只有城市
        if cityname != "" and airport == "":
            clear_cache(extra_data['token'], 'airport')
        # 只有机场车站
        if cityname == "" and airport != "":
            clear_cache(extra_data['token'], 'cityname')

        notchs = get_all_cache(extra_data['token'])
        logging.info('槽口检查: %s', notchs)
        if notchs:
            if 'city' not in notchs.keys() and 'airport' not in notchs.keys():
                return False, "请告诉我您咨询的城市或机场名称。"
            if not only_proprietary:
                if 'phone' not in notchs.keys():
                    return False, "您的会员注册手机号是多少？"
        return True, ""

    # 返回Statement结构
    def __return_response(self, msg, confidence=1):
        from chatterbot.conversation import Statement

        st = Statement(msg)
        st.confidence = confidence
        return st
