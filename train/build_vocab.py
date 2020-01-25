#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
build input/output vocabulary
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import Counter
import logging
import os
import sys

from khaiii.resource.vocabulary import VocabIn, VocabOut


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


#############
# functions #
#############
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
            _LOG.info('%dm-th line', line_num // 1000000)
        line = line.rstrip('\r\n')
        if not line:
            continue
        raw, tagged = line.split('\t')
        in_cnt.update(list(raw))
        out_cnt.update(tagged.split())

    os.makedirs(args.rsc_src, exist_ok=True)
    VocabIn.save(in_cnt, f'{args.rsc_src}/vocab.in', 2)
    VocabOut.save(out_cnt, f'{args.rsc_src}/vocab.out')


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='build input/output vocabulary')
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
