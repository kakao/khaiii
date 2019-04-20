#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
khaiii 출력 형태의 두 파일을 읽어들여 f-score를 측정
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import Counter
import logging
import sys
from typing import Iterator, Set, Tuple


#############
# functions #
#############
def _load(path: str) -> Iterator[Tuple[str, str]]:
    """
    파일을 읽어들여 (어절, 형태소)를 리턴하는 제너레이터
    Args:
        path:  file path
    Yields:
        word
        morphs
    """
    for line in open(path, 'r', encoding='UTF-8'):
        line = line.rstrip('\r\n')
        if not line:
            yield '', ''
            continue
        word, morphs = line.split('\t')
        yield word, morphs


def _morphs_to_set(morphs: str) -> Set[Tuple[str, int]]:
    """
    make set from morpheme string
    Args:
        morphs:  morpheme string
    Returns:
        morphemes set
    """
    morph_cnt = Counter([m for m in morphs.split(' + ')])
    morph_set = set()
    for morph, freq in morph_cnt.items():
        if freq == 1:
            morph_set.add(morph)
        else:
            morph_set.update([(morph, i) for i in range(freq)])
    return morph_set


def _count(cnt: Counter, gold: str, pred: str):
    """
    count gold and pred morphemes
    Args:
        cnt:  Counter object
        gold:  gold standard morphemes
        pred:  prediction morphemes
    """
    gold_set = _morphs_to_set(gold)
    pred_set = _morphs_to_set(pred)
    cnt['gold'] += len(gold_set)
    cnt['pred'] += len(pred_set)
    cnt['match'] += len(gold_set & pred_set)


def _report(cnt: Counter):
    """
    report metric
    Args:
        cnt:  Counter object
    """
    precision = 100 * cnt['match'] / cnt['pred']
    recall = 100 * cnt['match'] / cnt['gold']
    f_score = 2 * precision * recall / (precision + recall)
    print(f'precision: {precision:.2f}')
    print(f'recall: {recall:.2f}')
    print(f'f-score: {f_score:.2f}')


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    cnt = Counter()
    for line_num, (gold, pred) in enumerate(zip(_load(args.gold), _load(args.pred)), start=1):
        word_gold, morphs_gold = gold
        word_pred, morphs_pred = pred
        if word_gold != word_pred:
            raise ValueError(f'invalid align at {line_num}: {word_gold} vs {word_pred}')
        if not word_gold or not word_pred:
            continue
        _count(cnt, morphs_gold, morphs_pred)
    _report(cnt)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='command line part-of-speech tagger demo')
    parser.add_argument('-g', '--gold', help='gold standard file', metavar='FILE', required=True)
    parser.add_argument('-p', '--pred', help='prediction file', metavar='FILE', required=True)
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
