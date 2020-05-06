"""
从会员组获取用户的基础信息
"""
import requests
import hashlib
import time
import json


class SearchMemberInfo(object):
    jmember_url = ""

    def get_customer_hall_range(self, phone):
        """
        获取指定手机号用户所提供的休息厅id
        :return: 用户信息，及休息厅
        """
        return ""

    def get_cust_profile_describe(self, phone):
        """
        获取指定手机号用户的会员描述信息
        :param phone:
        :return:
        """
      
        return ""

    def _get_cust_service_scope(self, phone):
        """
        获取会员的服务范围
        """
        return ""

    def _get_cust_profile(self, jsjid):
        """
        获取用户概要信息
        """
        return ""
