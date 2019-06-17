#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
추출된 오분석 패치 후보를 검증하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import io
import logging
import os
import sys
from typing import Dict, Iterator, List, Tuple

from khaiii.khaiii import KhaiiiApi

from khaiii.munjong.sejong_corpus import Sentence, sents
from khaiii.resource.char_align import Aligner, AlignError, align_patch, align_to_tag
from khaiii.resource.morphs import mix_char_tag, WORD_DELIM_NUM, SENT_DELIM_NUM
from khaiii.resource.resource import load_vocab_out, parse_restore_dic


#########
# types #
#########
class StrFile(io.StringIO):
    """
    StringIO 객체에 name 멤버를 추가해 파일 객체인 것 처럼 동작하도록 하기 위해
    """
    def __init__(self, name, buf):
        """
        Args:
            name:  file name
            buf:  buffer
        """
        super().__init__(buf)
        self.name = name


#############
# variables #
#############
_CORPUS = []    # corpus cache


##########
# mapper #
##########
def _sent_iter(args: Namespace) -> Iterator[Sentence]:
    """
    sentence generator
    Args:
        args:  arguments
    Yields:
        sentence
    """
    if not _CORPUS:
        for name in sorted(os.listdir(args.corpus_dir)):
            if not name.endswith('.txt'):
                continue
            path = '{}/{}'.format(args.corpus_dir, name)
            _CORPUS.append((name, open(path, 'r', encoding='UTF-8').read()))
        _CORPUS.sort()

    for name, lines in _CORPUS:
        logging.info(name)
        for sent in sents(StrFile(name, lines)):
            yield sent


def _find_list(haystack: list, needle: list) -> int:
    """
    Return the index at which the sequence needle appears in the sequence haystack,
    or -1 if it is not found, using the Boyer-Moore-Horspool algorithm.
    The elements of needle and haystack must be hashable.

    >>> _find_list([10, 10, 20], [10, 20])
    1

    Args:
        haystack:  list to find
        needle:  pattern list
    Returns:
        start index. -1 if not found
    """
    h_len = len(haystack)
    n_len = len(needle)
    skip = {needle[i]: n_len - i - 1 for i in range(n_len - 1)}
    idx = n_len - 1
    while idx < h_len:
        for jdx in range(n_len):
            if haystack[idx - jdx] != needle[-jdx - 1]:
                idx += skip.get(haystack[idx], n_len)
                break
        else:
            return idx - n_len + 1
    return -1


def _align_sent(rsc_src: Tuple[Aligner, dict, Dict[str, str]], sent: Sentence) -> List[int]:
    """
    세종 문장 객체를 정렬하여 음절 별 출력 태그의 벡터로 표현한다.
    Args:
        rsc_src:  (Aligner, restore dic, vocab out) resource triple
        sent:  sejong_corpus.Sentence 객체
    Returns:
        list of output tag numbers. empty list for alignment error
    """
    aligner, restore_dic, vocab_out = rsc_src
    tag_nums = []
    restore_new = defaultdict(dict)
    vocab_new = defaultdict(list)
    for word in sent.words:
        try:
            word_align = aligner.align(word)
            _, word_tag_nums = align_to_tag(word.raw, word_align, (restore_dic, restore_new),
                                            (vocab_out, vocab_new))
        except AlignError as algn_err:
            logging.debug('alignment error: %s', word)
            logging.debug(str(algn_err))
            return []
        if tag_nums:
            tag_nums.append(WORD_DELIM_NUM)
        tag_nums.extend(word_tag_nums)
    tag_nums.insert(0, SENT_DELIM_NUM)
    tag_nums.append(SENT_DELIM_NUM)
    return tag_nums


def _analyze_sent(khaiii_api: KhaiiiApi, raw_sent: str) -> List[int]:
    """
    원시 문장에 대해 패치를 적용하지 않은 음절별 태깅 결과를 얻는다.
    Args:
        khaiii_api:  khaiii API 객체
        raw_sent:  원시 문장
    Returns:
        list of output tag numbers
    """
    tag_nums = khaiii_api.analyze_bfr_errpatch(raw_sent, '')
    logging.debug(tag_nums)
    return tag_nums


def _cnt_pos_neg(khaiii_api: KhaiiiApi, patch_raw: str, alignment: Tuple[list, list],
                 rsc_src: Tuple[Aligner, dict, Dict[str, str]], sent: Sentence) -> Tuple[int, int]:
    """
    오분석을 정분석으로 바꾼 횟수와, 오분석을 다른 오분석으로 바꾼 횟수를 센다.
    Args:
        khaiii_api:  khaiii API object
        patch_raw:  raw part of patch
        alignment:  (left, right) alignment pair
        rsc_src:  (Aligner, restore dic, vocab out) resource triple
        sent:  Sentence object
    Returns:
        오분석 -> 정분석 횟수
        오분석 -> 오분석 횟수
    """
    raw_sent = sent.raw_str()
    if patch_raw not in raw_sent:
        # 원문이 문장에서 발견되지 않으면 스킵
        return 0, 0
    aligner, restore_dic, vocab_out = rsc_src
    sent_align = _align_sent((aligner, restore_dic, vocab_out), sent)
    if not sent_align:
        # 코퍼스 정답이 원문과 정렬이 되지 않고 오류가 발생하면 스킵
        return 0, 0
    left_align, right_align = alignment
    left_needle = mix_char_tag(patch_raw, left_align)
    sent_anal = khaiii_api.analyze_bfr_errpatch(raw_sent, '')
    sent_haystack = mix_char_tag(raw_sent, sent_anal)
    pos_cnt = 0
    neg_cnt = 0
    found = _find_list(sent_haystack, left_needle)
    while found >= 0:
        # 패치의 좌측 오분석 열이 분석 결과에서 나타난 경우 우측 정답 열과 코퍼스를 비교
        right_corpus = sent_align[found:found + len(left_needle)]
        if right_align == right_corpus:
            pos_cnt += 1
        else:
            neg_cnt += 1
        del sent_haystack[:found + len(left_needle)]
        found = _find_list(sent_haystack, left_needle)
    return pos_cnt, neg_cnt


def run(args: Namespace):
    """
    actual function which is doing some task
    Args:
        args:  program arguments
    """
    aligner = Aligner(args.rsc_src)
    restore_dic = parse_restore_dic('{}/restore.dic'.format(args.rsc_src))
    if not restore_dic:
        sys.exit(1)
    vocab_out = load_vocab_out(args.rsc_src)

    khaiii_api = KhaiiiApi(args.lib_path, args.rsc_dir, '{"errpatch": false}')

    for line_num, line in enumerate(sys.stdin, start=1):
        line = line.rstrip('\r\n')
        if not line or line[0] == '#':
            continue
        raw, left, right = line.split('\t')
        left_align = align_patch((aligner, restore_dic, vocab_out), raw, left)
        if not left_align:
            logging.info('invalid %d-th line: left align: %s', line_num, line)
            continue
        right_align = align_patch((aligner, restore_dic, vocab_out), raw, right)
        if not right_align:
            logging.info('invalid %d-th line: right align: %s', line_num, line)
            continue
        if len(left_align) != len(right_align):
            logging.info('invalid %d-th line: left/right diff: %s', line_num, line)
            continue
        pos_cnt = 0
        neg_cnt = 0
        for sent in _sent_iter(args):
            pos_cnt_sent, neg_cnt_sent = _cnt_pos_neg(khaiii_api, raw, (left_align, right_align),
                                                      (aligner, restore_dic, vocab_out), sent)
            pos_cnt += pos_cnt_sent
            neg_cnt += neg_cnt_sent
            if neg_cnt > 0:
                break
        if neg_cnt > 0 or pos_cnt == 0:
            logging.info('invalid %d-th line: +%d, -%d: %s', line_num, pos_cnt, neg_cnt, line)
            continue
        print('{}\t{}\t{}'.format(raw, left, right))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='추출된 오분석 패치 후보를 검증하는 스크립트')
    parser.add_argument('-c', '--corpus-dir', help='corpus dir', metavar='DIR', required=True)
    parser.add_argument('--rsc-src', help='resource source dir <default: ../rsc/src>',
                        metavar='DIR', default='../rsc/src')
    parser.add_argument('--lib-path', help='khaiii shared library path', metavar='FILE', default='')
    parser.add_argument('--rsc-dir', help='resource dir', metavar='DIR', default='')
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

    run(args)


if __name__ == '__main__':
    main()
