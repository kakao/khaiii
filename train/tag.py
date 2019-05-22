#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
command line part-of-speech tagger demo
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import sys

from khaiii.train.tagger import PosTagger


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    tgr = PosTagger(args.model_dir, args.gpu_num)
    for line_num, line in enumerate(sys.stdin, start=1):
        if line_num % 100000 == 0:
            logging.info('%d00k-th line..', (line_num // 100000))
        line = line.rstrip('\r\n')
        if not line:
            print()
            continue
        pos_sent = tgr.tag_raw(line)
        for pos_word in pos_sent.pos_tagged_words:
            print(pos_word.raw, end='\t')
            print(' + '.join([str(m) for m in pos_word.pos_tagged_morphs]))
        print()


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='command line part-of-speech tagger demo')
    parser.add_argument('-m', '--model-dir', help='model dir', metavar='DIR', required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--gpu-num', help='GPU number to use <default: -1 for CPU>', metavar='INT',
                        type=int, default=-1)
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

    run(args)


if __name__ == '__main__':
    main()
