#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
recover wide char quotations in Sejong corpus
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

from khaiii.munjong.sejong_corpus import Word, WORD_ID_PTN


#############
# constants #
#############
_QUOT_NORM = {
    '"': '"',
    '“': '"',
    '”': '"',
    "'": "'",
    "‘": "'",
    "’": "'",
    "`": "'",
}


#############
# functions #
#############
def _recover(word: Word):
    """
    recover wide char quotations
    Args:
        word:  Word object
    """
    word_quots = [_ for _ in word.raw if _ in _QUOT_NORM]
    morph_quots = []
    for idx, morph in enumerate(word.morphs):
        if morph.tag != 'SS' or morph.lex not in _QUOT_NORM:
            continue
        morph_quots.append((idx, morph))
        quot_idx = len(morph_quots)-1
        if len(word_quots) <= quot_idx or _QUOT_NORM[word_quots[quot_idx]] != _QUOT_NORM[morph.lex]:
            logging.error('%d-th quots are different: %s', quot_idx+1, word)
            return
    if len(word_quots) != len(morph_quots):
        morph_quots = [_ for _ in word.morph_str() if _ in _QUOT_NORM]
        if word_quots != morph_quots:
            logging.error('number of quots are different: %s', word)
        return
    for word_char, (idx, morph) in zip(word_quots, morph_quots):
        if word_char == morph.lex:
            continue
        morph.lex = word_char


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
        _recover(word)
        print(word)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='recover wide char quotations in Sejong corpus')
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
