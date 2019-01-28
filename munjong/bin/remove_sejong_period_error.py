#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
remove wrong sentence breaking marks after period error eojeol
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2017-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser
import logging
import os
import re
import sys
from typing import TextIO, Tuple

from khaiii.munjong.sejong_corpus import Morph, WORD_ID_PTN


#############
# functions #
#############
def _get_three_lines(fin: TextIO) -> Tuple[str, str, str]:
    """
    get three lines tuple from file (generator)
    Args:
        fin:  input file
    Yields:
        prev. prev. line
        prev. line
        curr. line
    """
    prev_prev_line = fin.readline().rstrip('\r\n')
    prev_line = fin.readline().rstrip('\r\n')
    # print first two lines
    print(prev_prev_line)
    print(prev_line)
    for curr_line in fin:
        curr_line = curr_line.rstrip('\r\n')
        yield prev_prev_line, prev_line, curr_line
        prev_prev_line = prev_line
        prev_line = curr_line


def _is_known_period_error_eojeol(line: str) -> bool:
    """
    알려진 특정 문장분리 오류를 포함하는 어절인 지 여부
    Args:
        line:  line (eojeol)
    Returns:
        whether has error or not
    """
    cols = line.split('\t')
    if len(cols) != 3 or not WORD_ID_PTN.match(cols[0]):
        return False
    if '/SF + ' not in cols[2] or re.match(r'.+/EF \+ ./SF$', cols[2]):
        return False
    if re.match(r'.+/SF \+ [\'"’”]/SS$', cols[2]):
        return False
    morphs = [Morph.parse(_) for _ in cols[2].split(' + ')]
    tags_str = '+'.join([_.tag for _ in morphs])
    if 'SN+SF+SN' in tags_str and not tags_str.endswith('+SF'):
        # 4.6판: 4/SN + ./SF + 6/SN + 판/NNB
        if 'XSN+SF+SN' not in tags_str:
            return True
    elif 'SL+SF+SL' in tags_str and not tags_str.endswith('+SF'):
        # S.M.오너: S/SL + ./SF + M/SL + ./SF + 오너/NNG
        return True
    return False


def run():
    """
    run function which is the start point of program
    """
    file_name = os.path.basename(sys.stdin.name)
    for line_num, (prev_prev_line, prev_line, curr_line) in enumerate(_get_three_lines(sys.stdin),
                                                                      start=1):
        if curr_line == '</p>' and _is_known_period_error_eojeol(prev_line):
            continue
        elif prev_line == '</p>' and curr_line == '<p>' and \
                _is_known_period_error_eojeol(prev_prev_line):
            logging.info('%s:%d\t%s', file_name, line_num, prev_prev_line)
            continue
        print(curr_line)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='remove wrong sentence breaking marks after'
                                        ' period error eojeol')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'rt')
    if args.output:
        sys.stdout = open(args.output, 'wt')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()
