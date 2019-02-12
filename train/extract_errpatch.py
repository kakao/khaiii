#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
오분석 패치 후보를 추출하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import Counter
import difflib
import logging
import multiprocessing
import os
import sys
from typing import Iterator, List, Tuple

from khaiii.munjong import sejong_corpus
from khaiii.resource.morphs import WORD_DELIM_STR, SENT_DELIM_STR

from khaiii.khaiii import KhaiiiApi


#############
# variables #
#############
_KHAIII = None    # khaiii API


#############
# functions #
#############
def _get_diff_sgmts(result_morphs: List[str], corpus_morphs: List[str], raw_sent: str,
                    alignments: List[Tuple[int, int]]) -> List[Tuple[str, str, str]]:
    """
    패치 생성을 위해 자동 태깅 결과와 코퍼스의 정답이 다른 부분을 찾는다.
    Args:
        result_morphs:  자동 태깅 결과
        corpus_morphs:  코퍼스의 정답
        raw_sent:  문장 원문
        alignments:  음절별 정렬 정보. 형태소별 (시작, 끝) 음절 포지션 pair
    Returns:
        (자동, 정답, 원문) triple의 리스트
    """
    def _get_origin(first_morph: int, last_morph: int) -> str:
        """
        첫번째 형태소와 마지막 형태소 인덱스를 바탕으로 원문을 얻는다.
        Args:
            first_morph:  첫번째 형태소 인덱스
            last_morph:  마지막 형태소 인덱스
        Returns:
            원문
        """
        if first_morph < 0:
            first_morph = 0
        if last_morph >= len(alignments):
            last_morph = len(alignments)
        return raw_sent[alignments[first_morph][0]:alignments[last_morph-1][1]]

    def _expand_begin(result_begin: int) -> int:
        """
        왼쪽 경계가 음소로 나눠진 경우 음절 경계가 나올 때까지 왼쪽으로 확장한다.
        Args:
            result_begin:  왼쪽 경계
        Returns:
            확장된 왼쪽 경계
        """
        if result_begin <= 0:
            return result_begin
        if alignments[result_begin - 1][1] > alignments[result_begin][0]:
            return _expand_begin(result_begin - 1)
        return result_begin

    def _expand_end(result_end: int) -> int:
        """
        오른쪽 경계가 음소로 나눠진 경우 음절 경계가 나올 때까지 오른쪽으로 확장한다.
        Args:
            result_end:  오른쪽 경계.
                         [result_begin, result_end)으로 오른쪽 경계는 open이므로,
                         실제로 result_end-1이 마지막 형태소의 인덱스이다.
        Returns:
            확장된 오른쪽 경계
        """
        if result_end >= len(result_morphs):
            return result_end
        if alignments[result_end - 1][1] > alignments[result_end][0]:
            return _expand_end(result_end + 1)
        return result_end

    def _expand_spc(diff_sgmts: List[Tuple[str, str, str]], result_begin: int, result_end: int,
                    corpus_begin: int, corpus_end: int):
        """
        좌, 우 경계에 공백이 있을 경우 한 형태소 더 확장한다.
        Args:
            diff_sgmts:  diff를 추가할 리스트
            result_begin:  오분석의 왼쪽 경계
            result_end:  오분석의 오른쪽 경계
            corpus_begin:  정분석(코퍼스)의 왼쪽 경계
            corpus_begin:  정분석(코퍼스)의 오른쪽 경계
        """
        if (result_morphs[result_begin] != WORD_DELIM_STR and    # pylint: disable=consider-using-in
                result_morphs[result_end-1] != WORD_DELIM_STR):
            return
        _result_begin = _expand_begin(result_begin-1) \
            if result_morphs[result_begin] == WORD_DELIM_STR else result_begin
        result_end_ = _expand_end(result_end+1) \
            if result_morphs[result_end-1] == WORD_DELIM_STR else result_end
        assert result_begin != _result_begin or result_end != result_end_
        _left = result_morphs[_result_begin:result_end_]
        _corpus_begin = corpus_begin - (result_begin - _result_begin)
        corpus_end_ = corpus_end + (result_end_ - result_end)
        right_ = corpus_morphs[_corpus_begin:corpus_end_]
        origin = _get_origin(_result_begin, result_end_)
        diff_sgmts.append((origin, _left, right_))

    diff_sgmts = []
    matches = difflib.SequenceMatcher(None, result_morphs, corpus_morphs).get_matching_blocks()
    result_begin = 0
    corpus_begin = 0
    for match in matches:
        result_end = match.a
        left = []
        if result_end > result_begin:
            # _result_begin은 result_begin에서 왼쪽으로 같은 음절인 경우에 대해 확장한 시작 위치입니다.
            # 가령: '된다'에 대해 오분석: '되+ㄴ+다', 정분석: '되+ㄴ다'일 경우 틀린 영역은 'ㄴ+다' vs 'ㄴ다' 이지만,
            # 'ㄴ'이 포함된 음절이 '된'이므로 원문을 추출하면 '된'이 되어벼러 원문과 형태소간 불일치가 일어납니다.
            # 이것을 보정해 주기 위해 음소단위 형태소인 경우 왼쪽으로 음절 경계까지 확장하는 것입니다.
            # result_end_도 마찬가지로 result_end를 오른쪽으로 확장한 것입니다.
            _result_begin = _expand_begin(result_begin)
            result_end_ = _expand_end(result_end)
            left = result_morphs[_result_begin:result_end_]
        else:
            _result_begin = result_begin
            result_end_ = result_end

        corpus_end = match.b
        # _result_begin, result_end_가 각각 n, m개 확장된 경우,
        # 동일한 갯수로 corpus_begin, corpus_end을 좌, 우로 n, m개 확장합니다.
        # 틀린 부분의 좌, 우 형태소가 오분석과 정분석이 같다는 가정 하에 alignments 정보를 활용하지 않습니다.
        _corpus_begin = corpus_begin - (result_begin - _result_begin)
        corpus_end_ = corpus_end + (result_end_ - result_end)
        right = corpus_morphs[_corpus_begin:corpus_end_]

        if not left and not right:
            result_begin = result_end + match.size
            corpus_begin = corpus_end + match.size
            continue
        assert left != right, '{} == {}'.format(left, right)

        if (len(left) >= 2 and right) or (left and len(right) >= 2):
            # 결과가 다른 영역만
            origin = _get_origin(_result_begin, result_end_)
            diff_sgmts.append((origin, left, right))

        # _left는 left에서 오분석인 영역에 왼쪽으로 하나의 형태소를 더해서 만든 것입니다.
        # left_는 left에서 오른쪽으로 하나의 형태소, _left_는 양쪽에 하나씩 더해서 만든 것입니다.
        # _right, right_, _right_도 left와 마찬가지입니다.

        # 왼쪽 형태소를 추가
        # 추가한 형태소가 음소로 나눠진 경우에도 역시 왼쪽으로 음절 경계까지 확장
        __result_begin = _expand_begin(_result_begin-1)
        _left = result_morphs[__result_begin:_result_begin] + left
        __corpus_begin = _corpus_begin - (_result_begin - __result_begin)
        _right = corpus_morphs[__corpus_begin:_corpus_begin] + right
        left_first_lex = _left[0].rsplit('/', 1)[0]
        right_first_lex = _right[0].rsplit('/', 1)[0]
        if left_first_lex == right_first_lex and len(_left) >= 2 and len(_right) >= 2:
            origin = _get_origin(__result_begin, result_end_)
            diff_sgmts.append((origin, _left, _right))
            _expand_spc(diff_sgmts, __result_begin, result_end_, __corpus_begin, corpus_end_)

        # 오른쪽 형태소를 추가
        # 추가한 형태소가 음소로 나눠진 경우에도 역시 오른쪽으로 음절 경계까지 확장
        result_end__ = _expand_end(result_end_+1)
        left_ = left + result_morphs[result_end_:result_end__]
        corpus_end__ = corpus_end_ + (result_end__ - result_end_)
        right_ = right + corpus_morphs[corpus_end_:corpus_end__]
        left_last_lex = left_[-1].rsplit('/', 1)[0]
        right_last_lex = right_[-1].rsplit('/', 1)[0]
        if left_last_lex == right_last_lex and len(left_) >= 2 and len(right_) >= 2:
            origin = _get_origin(_result_begin, result_end__)
            diff_sgmts.append((origin, left_, right_))
            _expand_spc(diff_sgmts, _result_begin, result_end__, _corpus_begin, corpus_end__)

        # 양쪽에 형태소를 각각 추가
        _left_ = _left + result_morphs[result_end_:result_end__]
        _right_ = _right + corpus_morphs[corpus_end_:corpus_end__]
        origin = _get_origin(__result_begin, result_end__)
        diff_sgmts.append((origin, _left_, _right_))
        _expand_spc(diff_sgmts, __result_begin, result_end__, __corpus_begin, corpus_end__)

        result_begin = result_end + match.size
        corpus_begin = corpus_end + match.size
    return diff_sgmts


def _count_error(args: Namespace, doc_path: str) -> Counter:
    """
    count from courpus and make ambiguous dictionary
    Args:
        args:  program arguments
        doc_path:  document path
    Returns:
        오분석 패치 후보(원문, 오분석, 정분석 triple)의 카운터
    """
    global _KHAIII    # pylint: disable=global-statement
    if not _KHAIII:
        _KHAIII = KhaiiiApi(args.lib_path, args.rsc_dir)

    cnt = Counter()
    logging.info(doc_path)
    for sent in sejong_corpus.sents(open(doc_path, 'r', encoding='UTF-8')):
        raw_sent = sent.raw_str()
        result_morphs = [SENT_DELIM_STR, ]
        corpus_morphs = [SENT_DELIM_STR, ]
        alignments = [(0, 0), ]
        for result_word, corpus_word in zip(_KHAIII.analyze(raw_sent, ''), sent.words):
            assert result_word.lex == corpus_word.raw, \
                   '{}: "{}" != "{}"'.format(os.path.basename(doc_path), result_word, corpus_word)
            if len(result_morphs) > 1:
                result_morphs.append(WORD_DELIM_STR)
                corpus_morphs.append(WORD_DELIM_STR)
                alignments.append((alignments[-1][1], alignments[-1][1]+1))
            result_morphs.extend(['{}/{}'.format(m.lex, m.tag) for m in result_word.morphs])
            corpus_morphs.extend([str(m) for m in corpus_word.morphs])
            alignments.extend([(m.begin, m.begin + m.length) for m in result_word.morphs])
        result_morphs.append(SENT_DELIM_STR)
        corpus_morphs.append(SENT_DELIM_STR)
        alignments.append((alignments[-1][1], alignments[-1][1]))
        if result_morphs != corpus_morphs:
            diff_sgmts = _get_diff_sgmts(result_morphs, corpus_morphs, raw_sent, alignments)
            for origin, left, right in diff_sgmts:
                cnt[origin, ' + '.join(left), ' + '.join(right)] += 1
    return cnt


def _doc_iter(corpus_dir: str) -> Iterator[str]:
    """
    문종 코퍼스에서 문서의 경로를 리턴하는 generator
    Args:
        args:  program arguments
    Yields:
        document path
    """
    for name in sorted(os.listdir(corpus_dir)):
        if not name.endswith('.txt'):
            continue
        yield '{}/{}'.format(corpus_dir, name)


def _filter_cnt(args: Namespace, cnt: Counter) -> Counter:
    """
    규칙에 의해 필터링하여 빈도로 정렬한다.
    Args:
        args:  program arguments
        cnt:  (원문, left, right) 별로 측정한 빈도 사전
    Returns:
        (빈도, (원문, left, right)) pair의 리스트
    """
    cnts = []
    for (origin, left, right), freq in cnt.items():
        if freq < args.min_freq:
            continue
        if len(origin.strip()) < args.min_len:
            continue
        cnts.append((freq, (origin, left, right)))
    cnts.sort(key=lambda x: x[0], reverse=True)
    return cnts


def _print_cnt(args: Namespace, cnts: Counter):
    """
    최종 패치 후포를 출력한다.
    Args:
        args:  program arguments
        cnts:  counts
    """
    for freq, (origin, left, right) in cnts:
        freq_tab = ''
        if args.with_freq:
            freq_tab = '{}\t'.format(freq)
        print('{}{}\t{}\t{}'.format(freq_tab, origin, left, right))


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    cnt = Counter()
    if args.num_proc > 1:
        pool = multiprocessing.Pool(args.num_proc)
        results = ((doc, pool.apply_async(_count_error, (args, doc)))
                   for doc in _doc_iter(args.corpus_dir))
        for num, (doc, result) in enumerate(results, start=1):
            if num % 100 == 0:
                logging.info('%d-th document..', num)
            try:
                cnt.update(result.get(timeout=1000))
            except multiprocessing.context.TimeoutError:
                logging.error('timeout[%d]: %s', num, doc)
    else:
        for num, doc in enumerate(_doc_iter(args.corpus_dir), start=1):
            if num % 10 == 0:
                logging.info('%d-th document..', num)
            cnt.update(_count_error(args, doc))

    cnts = _filter_cnt(args, cnt)
    _print_cnt(args, cnts)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='오분석 패치 후보를 추출하는 스크립트')
    parser.add_argument('-c', '--corpus-dir', help='corpus dir', metavar='DIR', required=True)
    parser.add_argument('--lib-path', help='khaiii shared library path' \
                                           ' <default ../build/lib/libkhaiii.so>',
                        metavar='FILE', default='../build/lib/libkhaiii.so')
    parser.add_argument('--rsc-dir', help='resource dir <default: ../build/share/khaiii>',
                        metavar='DIR', default='../build/share/khaiii')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--with-freq', help='print with frequency', action='store_true')
    parser.add_argument('--min-freq', help='minimum frequency <default: 10>', metavar='FREQ',
                        type=int, default=10)
    parser.add_argument('--min-len', help='minimum original text length <default: 4>',
                        metavar='LEN', type=int, default=4)
    parser.add_argument('--num-proc', help='number of processes <default: 1>', metavar='NUM',
                        type=int, default=1)
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
