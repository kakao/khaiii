#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
detect period error of Sejong corpus
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser
import logging
import os
import re
import sys
from typing import Iterator, TextIO, Tuple

from khaiii.munjong.sejong_corpus import Morph, WORD_ID_PTN


#############
# functions #
#############
def _get_two_lines(fin: TextIO) -> Iterator[Tuple[str, str]]:
    """
    get two lines tuple from file (generator)
    Args:
        fin:  input file
    Yields:
        current line
        next line
    """
    curr_line = fin.readline().rstrip('\r\n')
    for next_line in fin:
        next_line = next_line.rstrip('\r\n')
        yield curr_line, next_line
        curr_line = next_line


def _is_correct_eos(line: str) -> bool:
    """
    whether correct end of sentence or not
    Args:
        line:  line (word)
    Returns:
        whether correct or not
    """
    _, _, morphs_str = line.split('\t')
    if re.match(r'.+/EF \+ ./SF$', morphs_str):
        return True
    if re.match(r'.+/SF \+ [\'"’”」\]]/SS$', morphs_str):
        return True
    morphs = [Morph.parse(_) for _ in morphs_str.split(' + ')]
    tags_str = '+'.join([_.tag for _ in morphs])
    if tags_str.endswith('+SF+SS+JKQ') or tags_str.endswith('+SF+SS+VCP+ETM'):
        return True
    return False


def run():
    """
    run function which is the start point of program
    """
    file_name = os.path.basename(sys.stdin.name)
    for line_num, (curr_line, next_line) in enumerate(_get_two_lines(sys.stdin), start=1):
        cols = curr_line.split('\t')
        if len(cols) != 3 or not WORD_ID_PTN.match(cols[0]):
            continue
        if '/SF + ' not in cols[2] or not next_line.startswith('</'):
            continue
        if _is_correct_eos(curr_line):
            continue
        print('{}:{}\t{}'.format(file_name, line_num, curr_line))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='detect period error of Sejong corpus')
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
