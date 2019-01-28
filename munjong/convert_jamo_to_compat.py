#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
한글 자모 영역의 코드를 호환 영역으로 변환
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser
import logging
import sys

from khaiii.munjong.sejong_corpus import WORD_ID_PTN
from khaiii.resource.jaso import norm_compat


#############
# functions #
#############
def _norm(text: str) -> str:
    """
    정규화를 수행하는 함수
    Args:
        text:  입력 텍스트
    Returns:
        정규화된 텍스트
    """
    normalized = norm_compat(text)
    normalized = normalized.replace('ᆞ', 'ㆍ')    # 0x119e -> 0x318d
    normalized = normalized.replace('ᄝ', 'ㅱ')    # 0x111d -> 0x3171
    return normalized


def run():
    """
    run function which is the start point of program
    """
    for line in sys.stdin:
        line = line.rstrip('\r\n')
        if not WORD_ID_PTN.match(line):
            print(line)
            continue
        wid, word, morph = line.split('\t')
        print('{}\t{}\t{}'.format(wid, _norm(word), _norm(morph)))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='한글 자모 영역의 코드를 호환 영역으로 변환')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()
