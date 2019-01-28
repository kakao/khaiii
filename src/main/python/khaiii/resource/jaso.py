# -*- coding: utf-8 -*-


"""
한글 자소 관련 유틸리티 모듈
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from typing import Tuple


#############
# constants #
#############
# 한글 자모 호환 영역 (초성과 종성이 같음. 두벌식 키보드로 입력할 때 들어가는 코드)
_FIRST_COMPAT = ['\u3131', '\u3132', '\u3134', '\u3137', '\u3138',    # 초성
                 '\u3139', '\u3141', '\u3142', '\u3143', '\u3145',
                 '\u3146', '\u3147', '\u3148', '\u3149', '\u314a',
                 '\u314b', '\u314c', '\u314d', '\u314e']
_MIDDLE_COMPAT = ['\u314f', '\u3150', '\u3151', '\u3152', '\u3153',    # 중성
                  '\u3154', '\u3155', '\u3156', '\u3157', '\u3158',
                  '\u3159', '\u315a', '\u315b', '\u315c', '\u315d',
                  '\u315e', '\u315f', '\u3160', '\u3161', '\u3162',
                  '\u3163']
_LAST_COMPAT = ['\u3131', '\u3132', '\u3133', '\u3134', '\u3135',    # 종성
                '\u3136', '\u3137', '\u3139', '\u313a', '\u313b',
                '\u313c', '\u313d', '\u313e', '\u313f', '\u3140',
                '\u3141', '\u3142', '\u3144', '\u3145', '\u3146',
                '\u3147', '\u3148', '\u314a', '\u314b', '\u314c',
                '\u314d', '\u314e']
_ALL_COMPAT = _FIRST_COMPAT + _MIDDLE_COMPAT + _LAST_COMPAT

# 한글 자모 영역 (초성과 종성이 다름. 세종 코퍼스에서 사용한 코드)
_FIRST_JAMO = ['\u1100', '\u1101', '\u1102', '\u1103', '\u1104',    # 초성
               '\u1105', '\u1106', '\u1107', '\u1108', '\u1109',
               '\u110a', '\u110b', '\u110c', '\u110d', '\u110e',
               '\u110f', '\u1110', '\u1111', '\u1112']
_MIDDLE_JAMO = ['\u1161', '\u1162', '\u1163', '\u1164', '\u1165',    # 중성
                '\u1166', '\u1167', '\u1168', '\u1169', '\u116a',
                '\u116b', '\u116c', '\u116d', '\u116e', '\u116f',
                '\u1170', '\u1171', '\u1172', '\u1173', '\u1174',
                '\u1175']
_LAST_JAMO = ['\u11a8', '\u11a9', '\u11aa', '\u11ab', '\u11ac',    # 종성
              '\u11ad', '\u11ae', '\u11af', '\u11b0', '\u11b1',
              '\u11b2', '\u11b3', '\u11b4', '\u11b5', '\u11b6',
              '\u11b7', '\u11b8', '\u11b9', '\u11ba', '\u11bb',
              '\u11bc', '\u11bd', '\u11be', '\u11bf', '\u11c0',
              '\u11c1', '\u11c2']
_ALL_JAMO = _FIRST_JAMO + _MIDDLE_JAMO + _LAST_JAMO
_ALL_JAMO_SET = set(_ALL_JAMO)
_JAMO_TO_COMPAT = dict(zip(_ALL_JAMO, _ALL_COMPAT))

# 반각 자모 영역 (호환 영역과 비슷하게 초성과 종성이 같으나 글자 폭이 절반인 코드)
_FIRST_HALFWIDTH = ['\uffa1', '\uffa2', '\uffa4', '\uffa7', '\uffa8',    # 초성
                    '\uffa9', '\uffb1', '\uffb2', '\uffb3', '\uffb5',
                    '\uffb6', '\uffb7', '\uffb8', '\uffb9', '\uffba',
                    '\uffbb', '\uffbc', '\uffbd', '\uffbe']
_MIDDLE_HALFWIDTH = ['\uffc2', '\uffc3', '\uffc4', '\uffc5', '\uffc6',    # 중성
                     '\uffc7', '\uffca', '\uffcb', '\uffcc', '\uffcd',
                     '\uffce', '\uffcf', '\uffd2', '\uffd3', '\uffd4',
                     '\uffd5', '\uffd6', '\uffd7', '\uffda', '\uffdb',
                     '\uffdc']
_LAST_HALFWIDTH = ['\uffa1', '\uffa2', '\uffa3', '\uffa4', '\uffa5',    # 종성
                   '\uffa6', '\uffa7', '\uffa9', '\uffaa', '\uffab',
                   '\uffac', '\uffad', '\uffae', '\uffaf', '\uffb0',
                   '\uffb1', '\uffb2', '\uffb4', '\uffb5', '\uffb6',
                   '\uffb7', '\uffb8', '\uffba', '\uffbb', '\uffbc',
                   '\uffbd', '\uffbe']
_ALL_HALFWIDTH = _FIRST_HALFWIDTH + _MIDDLE_HALFWIDTH + _LAST_HALFWIDTH
_ALL_HALFWIDTH_SET = set(_ALL_HALFWIDTH)
_HALFWIDTH_TO_COMPAT = dict(zip(_ALL_HALFWIDTH, _ALL_COMPAT))


#############
# functions #
#############
def _decomp_char(char: str) -> Tuple[str, str, str]:
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

    first = _FIRST_COMPAT[first_idx]
    middle = _MIDDLE_COMPAT[middle_idx]
    if not last_idx:
        return first, middle

    last = _LAST_COMPAT[last_idx-1]
    return first, middle, last


def decompose(text: str) -> str:
    """
    유니코드 한글 텍스트를 한글 호환영역 자소로 분해한다.
    Args:
        text:  한글 텍스트
    Returns:
        자소 분해된 텍스트
    """
    if not text:
        return text

    decomposed = []
    for char in text:
        code = ord(char)
        if code < 0xAC00 or code > 0xD7A3:
            decomposed.append(char)
        else:
            decomposed.extend(_decomp_char(char))
    return ''.join(decomposed)


def norm_compat(text: str) -> str:
    """
    유니코드 내 한글 자소를 호환 영역으로 정규화한다.
    Args:
        text:  한글 텍스트
    Returns:
        자소가 호환 영역으로 정규화된 텍스트
    """
    if not text:
        return text

    normalized = []
    for char in text:
        if char in _ALL_JAMO_SET:
            normalized.append(_JAMO_TO_COMPAT[char])
        elif char in _ALL_HALFWIDTH_SET:
            normalized.append(_HALFWIDTH_TO_COMPAT[char])
        else:
            normalized.append(char)
    return ''.join(normalized)
