import logging
import requests


class SearchVipHall:
    """
    休息厅数据问答
    """
    url = ''

    # 搜索所有自营厅
    def proprietary_hall(self, city_name='', air_port_name=''):
        return ""

    # 带用户身份信息搜索休息厅并生成标准应答话术
    def vip_hall_with_customer_info(self, customer_interests, city_name='', air_port_name=''):
        return ""

    # 实时获取休息厅的全部数据
    def get_all_vip_hall(self, city_name='', air_port_name=''):
        return ""

    # 根据休息厅编号获取休息厅信息
    def get_cooperation_vip_hall(self, vip_hall_list, city_name=None, air_port_name=None):
        return ""
