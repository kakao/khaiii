#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
코퍼스를 train/dev/test로 분할한다.
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import random
import sys
from typing import Iterator, List, TextIO


#############
# functions #
#############
def _sents(fin: TextIO) -> Iterator[List[str]]:
    """
    read from file and yield a sentence (generator)
    Args:
        fin:  input file
    Yields:
        sentence (list of lines)
    """
    sent = []
    for line in fin:
        line = line.rstrip('\r\n')
        if not line:
            if sent:
                yield sent
                sent = []
            continue
        sent.append(line)
    if sent:
        yield sent


def _write_to_file(path: str, sents: List[List[str]]):
    """
    파일에 쓴다.
    Args:
        path:  path
        sents:  sentences
    """
    with open(path, 'w', encoding='UTF-8') as fout:
        for sent in sents:
            print('\n'.join(sent), file=fout)
            print(file=fout)


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    sents = []
    for num, sent in enumerate(_sents(sys.stdin), start=1):
        if num % 100000 == 0:
            logging.info('%d00k-th sent..', num // 100000)
        sents.append(sent)
    random.shuffle(sents)
    _write_to_file('{}.dev'.format(args.out_pfx), sents[:args.dev])
    _write_to_file('{}.test'.format(args.out_pfx), sents[args.dev:args.dev+args.test])
    _write_to_file('{}.train'.format(args.out_pfx), sents[args.dev+args.test:])
    logging.info('dev / test / train: %d / %d / %d', args.dev, args.test,
                 len(sents[args.dev+args.test:]))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='코퍼스를 train/dev/test로 분할한다.')
    parser.add_argument('-o', '--out-pfx', help='output file prefix', metavar='NAME', required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--dev', help='number of sentence in dev set', metavar='NUM', type=int,
                        default=5000)
    parser.add_argument('--test', help='number of sentence in test set', metavar='NUM', type=int,
                        default=5000)
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
