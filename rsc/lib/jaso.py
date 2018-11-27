# -*- coding: utf-8 -*-


"""
한글 자소 관련 유틸리티 모듈
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


#############
# constants #
#############
_FIRST = ['\u3131', '\u3132', '\u3134', '\u3137', '\u3138',    # 초성
          '\u3139', '\u3141', '\u3142', '\u3143', '\u3145',
          '\u3146', '\u3147', '\u3148', '\u3149', '\u314a',
          '\u314b', '\u314c', '\u314d', '\u314e']
_MIDDLE = ['\u314f', '\u3150', '\u3151', '\u3152', '\u3153',    # 중성
           '\u3154', '\u3155', '\u3156', '\u3157', '\u3158',
           '\u3159', '\u315a', '\u315b', '\u315c', '\u315d',
           '\u315e', '\u315f', '\u3160', '\u3161', '\u3162',
           '\u3163']
_LAST = ['\u3131', '\u3132', '\u3133', '\u3134', '\u3135',    # 종성
         '\u3136', '\u3137', '\u3139', '\u313a', '\u313b',
         '\u313c', '\u313d', '\u313e', '\u313f', '\u3140',
         '\u3141', '\u3142', '\u3144', '\u3145', '\u3146',
         '\u3147', '\u3148', '\u314a', '\u314b', '\u314c',
         '\u314d', '\u314e']


#############
# functions #
#############
def _decomp_char(char):
    """
    한글 음절 하나를 자소로 분해한다.
    Args:
        char:  한글 음절
    Return:
        (초성, 중성, 종성) tuple
    """
    assert len(char) == 1, '입력한 문자열의 길이가 1이 아닙니다.'
    assert 0xAC00 <= ord(char) <= 0xD7A3, '입력 문자가 자소 분해가 가능한 한글 영역이 아닙니다.'

    idx = ord(char) - 0xAC00
    last_idx = idx % 28
    first_start = (idx - last_idx) // 28
    first_idx = first_start // 21
    middle_idx = first_start % 21

    first = _FIRST[first_idx]
    middle = _MIDDLE[middle_idx]
    if not last_idx:
        return first, middle

    last = _LAST[last_idx-1]
    return first, middle, last


def decompose(text: str) -> str:
    """
    유니코드 한글 텍스트를 한글 호환영역 자소로 분해한다.
    Args:
        text:  한글 텍스트
    Returns:
        자소 분해된 텍스트
    """
    decomposed = []
    for char in text:
        code = ord(char)
        if code < 0xAC00 or code > 0xD7A3:
            decomposed.append(char)
        else:
            decomposed.extend(_decomp_char(char))
    return ''.join(decomposed)
