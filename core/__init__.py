from enum import Enum, unique


@unique
class IntentionType(Enum):
    HallRoom = 0  # 休息厅
    VipCard = 1  # 会员卡
    AirTicket = 2  # 机票
    Other = 3  # 其他问题
