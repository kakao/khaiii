#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
command line part-of-speech tagger demo
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import sys

from khaiii.train.dataset import SentIter
from khaiii.train.tagger import PosTagger
from khaiii.train import dataset


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
    tgr = PosTagger(args.model_dir, args.gpu_num)
    data = dataset.load_raw(sys.stdin, tgr.cfg.window, tgr.rsc)    # pylint: disable=no-member

    device = f'cuda:{args.gpu_num}' if args.gpu_num >= 0 else 'cpu'
    for idx, (batch, sent) in enumerate(SentIter(data, device=device), start=1):
        if idx % 100000 == 0:
            _LOG.info('%d00k-th sentence..', (idx // 100000))
        tgr.tag_batch(batch, sent)
        for word in sent.words:
            print(word.raw, end='\t')
            print(' + '.join([str(m) for m in word.morphs]))
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
