import re


def extract_phone_numbers(string):
    """
    提取手机号, 若是语句中有多个匹配项，则返回空
    :param string:
    :return:
    """
    m = re.findall(r"1\d{10}", string)
    return m[0] if len(m) == 1 else ""


if __name__ == "__main__":
    print(extract_phone_numbers("1a312011111111111139"))
