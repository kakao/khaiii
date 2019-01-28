#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
recover cases of English letters in Sejong corpus
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser
import copy
import logging
import os
import re
import sys

from khaiii.munjong.sejong_corpus import Word, WORD_ID_PTN


#############
# functions #
#############
def _recover(word: Word):
    """
    recover cases
    Args:
        word:  Word object
    """
    word_letters = [_ for _ in word.raw if re.match(r'[a-zA-Z]', _)]
    letter_idx = -1
    is_recovered = False
    word_copy = copy.deepcopy(word)
    for morph in word_copy.morphs:
        for idx, char in enumerate(morph.lex):
            if not re.match(r'[a-zA-Z]', char):
                continue
            letter_idx += 1
            if word_letters[letter_idx] == char:
                continue
            morph.lex = morph.lex[:idx] + word_letters[letter_idx] + morph.lex[idx+1:]
            is_recovered = True
    if is_recovered:
        logging.info('%s  =>  %s', str(word), word_copy.morph_str())
        word.morphs = word_copy.morphs


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
        try:
            _recover(word)
        except IndexError as idx_err:
            logging.error('%s(%d): %s: %s', file_name, line_num, idx_err, word)
        print(word)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='recover cases of English letters in Sejong corpus')
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
