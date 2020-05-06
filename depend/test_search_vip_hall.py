from unittest import TestCase

from depend.server_vip_hall import SearchVipHall


class TestSearchVipHall(TestCase):
    def setUp(self):
        print()
        self.search = SearchVipHall()

    def test_get_all_vip_hall(self):
        data = self.search.proprietary_hall("哈尔滨")
        print(data)

    def test_get_cooperation_vip_hall(self):
        vip_hall_ids = [433, 432, 359, 323]
        data = self.search.get_cooperation_vip_hall(vip_hall_ids, '太原', '')
        print(data)
