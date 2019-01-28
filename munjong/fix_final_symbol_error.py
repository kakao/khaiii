#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
fix final symbol errors on Sejong corpus
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser
import logging
import os
import sys

from khaiii.munjong.sejong_corpus import Morph, Word, WORD_ID_PTN


#############
# functions #
#############
def _attach_missing_symbol(word: Word):
    """
    attach missing symbol
    Args:
        word:  Word object
    """
    raw_word = word.raw
    raw_morph = ''.join([_.lex for _ in word.morphs])
    if not raw_word.startswith(raw_morph) or len(raw_word) != len(raw_morph)+1:
        return
    last_symbol = raw_word[-1]
    if last_symbol == '.' and word.morphs[-1].tag == 'EC':
        word.morphs.append(Morph('.', 'SF'))
    elif last_symbol == ',':
        word.morphs.append(Morph(',', 'SP'))
    elif last_symbol == '"':
        word.morphs.append(Morph('"', 'SS'))


def run():
    """
    run function which is the start point of program
    """
    file_name = os.path.basename(sys.stdin.name)
    for line_num, line in enumerate(sys.stdin, start=1):
        line = line.rstrip('\r\n')
        if not WORD_ID_PTN.match(line):
            print(line)
            continue
        word = Word.parse(line, file_name, line_num)
        _attach_missing_symbol(word)
        print(word)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='fix final symbol errors on Sejong corpus')
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
