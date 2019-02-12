#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
입력(음절) 및 출력(태그) vocabulary를 생성한다.
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import Counter
import logging
import os
import sys
from typing import TextIO

from khaiii.resource.morphs import TAGS


#############
# functions #
#############
def _print(cnt: Counter, fout: TextIO, is_with_freq: bool = True):
    """
    vocabulary 사전을 출력한다.
    Args:
        cnt:  Counter object
        fout:  출력 파일
        is_with_freq:  빈도를 함께 출력할 지 여부
    """
    for char, freq in sorted(cnt.items(), key=lambda x: x[0]):
        if is_with_freq and freq < 2:
            continue
        if is_with_freq:
            print('{}\t{}'.format(char, freq), file=fout)
        else:
            print(char, file=fout)


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    in_cnt = Counter()
    out_cnt = Counter()
    for line_num, line in enumerate(sys.stdin, start=1):
        if line_num % 1000000 == 0:
            logging.info('%dm-th line', line_num // 1000000)
        line = line.rstrip('\r\n')
        if not line:
            continue
        raw, tagged = line.split('\t')
        in_cnt.update(list(raw))
        out_cnt.update([tag for tag in tagged.split() if tag[2:] not in TAGS])
    os.makedirs(args.rsc_src, exist_ok=True)
    with open('{}/vocab.in'.format(args.rsc_src), 'w', encoding='UTF-8') as fout:
        _print(in_cnt, fout)
    with open('{}/vocab.out'.format(args.rsc_src), 'w', encoding='UTF-8') as fout:
        print('\n'.join(['B-{}'.format(tag) for tag in TAGS]), file=fout)
        print('\n'.join(['I-{}'.format(tag) for tag in TAGS]), file=fout)
        _print(out_cnt, fout, is_with_freq=False)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='입력(음절) 및 출력(태그) vocabulary를 생성한다.')
    parser.add_argument('--rsc-src', help='resource source dir <default: ../rsc/src>',
                        metavar='DIR', default='../rsc/src')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
