from unittest import TestCase
from depend.server_member_info import SearchMemberInfo


class TestSearchMemberInfo(TestCase):

    def test_get_customer_info(self):
        response_json = SearchMemberInfo().get_customer_hall_range("1888888888")
        print(response_json)

    def test_get_cust_asset(self):
        response_json = SearchMemberInfo().get_cust_profile_describe("1777777777")
        print(response_json)
