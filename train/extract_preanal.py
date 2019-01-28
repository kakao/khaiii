#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
기분석 사전 후보를 추출하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
import logging
import sys
from typing import Dict, List, Tuple

from khaiii.munjong.sejong_corpus import Word
from khaiii.resource.char_align import Aligner, AlignError

import map_char_to_tag


#########
# types #
#########
class Entry:
    """
    preanalized dictionary entry
    """
    def __init__(self, freq: int, word: str, morph_str: str):
        self.is_del = False
        self.is_pfx = word[-1] == '\1'
        self.freq = freq
        self.word = word[:-1] if self.is_pfx else word
        self.morph_str = morph_str

    def __str__(self):
        return '{}{}\t{}{}\t{}'.format('-' if self.is_del else '', self.freq, self.word,
                                       '*' if self.is_pfx else '', self.morph_str)


#############
# variables #
#############
DIC_AMBIG = defaultdict(Counter)


#############
# functions #
#############
def _get_prefix(word: Word) -> Tuple[str, str]:
    """
    맨 뒤의 형태소 하나를 제외한 prefix와 그 분석 결과
    Args:
        word:  어절 객체
    Returns:
        prefix
        분석 결과
    """
    if len(word.morphs) < 2:
        return None, None
    if not word.raw.endswith(word.morphs[-1].lex):
        return None, None
    prefix = '{}\1'.format(word.raw[:-len(word.morphs[-1].lex)])
    if len(prefix) < 5:    # 4음절 미만은 버린다.
        return None, None
    morphs = ['{}/{}'.format(m.lex, m.tag) for m in word.morphs[:-1]]
    return prefix, ' + '.join(morphs)


def _count_ambig(args: Namespace):
    """
    count from courpus and make ambiguous dictionary
    Args:
        args:  program arguments
    """
    for num, sent in enumerate(map_char_to_tag.sent_iter(args), start=1):
        if num % 10000 == 0:
            logging.info('%dk-th sentence..', num // 1000)
        for word in sent.words:
            DIC_AMBIG[word.raw][word.morph_str()] += 1
            prefix, morph_pfx = _get_prefix(word)
            if prefix:
                DIC_AMBIG[prefix][morph_pfx] += 1


def _filter_no_ambig(min_freq: int) -> Dict[str, str]:
    """
    문맥과 상관 없이 중의성이 없는 엔트리를 출력
    Args:
        min_freq:  최소 빈도
    Returns:
        중의성이 있는 엔트리가 제거된 사전
    """
    dic_no_ambig = {}
    for word, cnt in DIC_AMBIG.items():
        if len(cnt) > 1:
            continue
        morph_str, freq = cnt.most_common(1)[0]
        if freq < min_freq:
            continue
        dic_no_ambig[word] = morph_str
    return dic_no_ambig


def _make_entries(dic_no_ambig: Dict[str, str]) -> List[Entry]:
    """
    기분석 사전 엔트리를 생성한다.
    Args:
        dic_no_ambig:  중의성이 없는 엔트리 사전
    Returns:
        엔트리 리스트
    """
    entries = []
    pfx_idx = -1
    for idx, (word, morph_str) in enumerate(sorted(dic_no_ambig.items())):
        _, freq = list(DIC_AMBIG[word].items())[0]
        entry = Entry(freq, word, morph_str)
        if entry.is_pfx:
            if entries and entries[-1].word == entry.word:
                # 이전 어절 exact가 현재 prefix와 같다면, 예: "제이미" vs "제이미*"
                if entries[-1].morph_str == entry.morph_str:    # pylint: disable=simplifiable-if-statement
                    # 분석 결과가 같다면 prefix를 남기고 exact를 제거한다.
                    entries[-1].is_del = True
                else:
                    # 분석 결과가 다르면 안전하게 exact를 남기고 prefix를 제거한다.
                    entry.is_del = True
            pfx_idx = idx
        elif (pfx_idx >= 0 and word.startswith(entries[pfx_idx].word) and
              morph_str.startswith(entries[pfx_idx].morph_str) and
              entries[pfx_idx].freq <= freq):
            # 이전 prefix가 현재 어절 exact를 포함하면서 빈도는 같다면,
            # 모두 동일한 어절에서 뽑힌 prefix이므로 prefix를 삭제한다.
            # 예: 빈도가 같은 "강화된다*"(삭제) vs "강화된다."(남김)
            entries[pfx_idx].is_del = True
        entries.append(entry)
    return entries


def run(args: Namespace):
    """
    map characters with tags from eojeol and morphemes
    Args:
        args:  program arguments
    """
    aligner = Aligner(args.rsc_src)

    _count_ambig(args)
    dic_no_ambig = _filter_no_ambig(args.min_freq)
    entries = _make_entries(dic_no_ambig)

    del_word = 0
    del_pfx = 0
    for entry in entries:
        if entry.is_del:
            logging.debug(entry)
            if entry.is_pfx:
                del_pfx += 1
            else:
                del_word += 1
        else:
            word = Word.parse('\t'.join(['', entry.word, entry.morph_str]), '', 0)
            try:
                aligner.align(word)
                entry_str = str(entry)
                if not args.with_freq:
                    entry_str = '\t'.join(entry_str.split('\t')[1:])
                print(entry_str)
            except AlignError as algn_err:
                logging.error('%s: %s', algn_err, entry)

    logging.info('deleted word: %d', del_word)
    logging.info('deleted prefix: %d', del_pfx)


########
# main #
########
def main():
    """
    map characters with tags from eojeol and morphemes
    """
    parser = ArgumentParser(description='기분석 사전 후보를 추출하는 스크립트')
    parser.add_argument('-c', '--corpus-dir', help='corpus dir', metavar='DIR', required=True)
    parser.add_argument('--rsc-src', help='resource source dir <default: ../rsc/src>',
                        metavar='DIR', default='../rsc/src')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--min-freq', help='minimum frequency <default: 10>', metavar='NUM',
                        type=int, default=10)
    parser.add_argument('--with-freq', help='print with frequency', action='store_true')
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
