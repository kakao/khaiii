#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
기분석 사전을 빌드하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import glob
import logging
import os
import struct
import sys
from typing import Dict, List

from khaiii.munjong import sejong_corpus
from khaiii.resource.char_align import Aligner, AlignError, align_to_tag
from khaiii.resource.morphs import Morph, ParseError
from khaiii.resource.resource import load_restore_dic, load_vocab_out
from khaiii.resource.trie import Trie

from compile_restore import append_new_entries



#########
# types #
#########
class Entry:
    """
    pre-analyzed dictionary entry
    """
    def __init__(self, file_path: str, line_num: int, line: str):
        """
        Args:
            file_path:  파일 경로
            line_num:  라인 번호
            line:  라인 원문
        """
        self.file_name = os.path.basename(file_path)
        self.line_num = line_num
        self.line = line
        self.err_msg = ''    # (엔트리에 에러가 존재할 경우) 에러 메세지
        self.is_sharp = False    # 샵(주석) 여부
        self.is_pfx = False    # 전방매칭 패턴 여부
        self.word = ''    # 어절 원문
        self.morphs = []    # 형태소 분석 결과
        self.align = []    # 음절단위 정렬 정보
        self.tag_outs = []    # 출력 태그 문자열
        self.tag_nums = []    # 출력 태그 숫자 (최종)
        self._parse()

    def __str__(self):
        file_num = '{}:{}: '.format(self.file_name, self.line_num) if self.file_name else ''
        line = '# {}'.format(self.line) if self.is_sharp else self.line
        if self.err_msg:
            return '{}{}: "{}"'.format(file_num, self.err_msg, line)
        if self.is_sharp:
            return '{}: "{}"'.format(file_num, line)
        return '{}{}\t{}'.format(self.word, '*' if self.is_pfx else '', Morph.to_str(self.morphs))

    def _parse(self):
        """
        기분석 사전 한 라인을 파싱한다.
        """
        if len(self.line) >= 2 and self.line.startswith('# '):
            self.is_sharp = True
            self.line = self.line[2:]
        cols = self.line.split('\t')
        if len(cols) != 2:
            if not self.is_sharp:
                self.err_msg = '[PARSE] number of columns must be 2, not {}'.format(len(cols))
            return
        self.word, morph_str = [c.strip() for c in cols]
        if not self.word:
            self.err_msg = '[PARSE] no word'
            return
        if len(self.word) > 2 and self.word[-1] == '*':
            self.is_pfx = True
            self.word = self.word[:-1].strip()
        elif ' ' in self.word:
            self.err_msg = '[PARSE] space in word'
            return
        try:
            self.morphs = Morph.parse(morph_str)
        except ParseError as par_err:
            self.err_msg = '[PARSE] {}'.format(par_err)


#############
# functions #
#############
def print_errors(entries: List[Entry]):
    """
    에러가 발생한 엔트리를 출력하고 프로그램을 종료한다.
    Args:
        entries:  엔트리 리스트
    """
    if entries:
        for entry in entries:
            logging.error(entry)
        logging.error('%d errors', len(entries))
        sys.exit(1)


def _load_entries(args: Namespace) -> List[Entry]:
    """
    사전 엔트리를 파일로부터 로드한다.
    Args:
        args:  arguments
    Returns:
        엔트리 리스트
    """
    good_entries = []
    bad_entries = []
    for file_path in glob.glob('{}/preanal.*'.format(args.rsc_src)):
        file_name = os.path.basename(file_path)
        logging.info(file_name)
        for line_num, line in enumerate(open(file_path, 'r', encoding='UTF-8'), start=1):
            line = line.rstrip('\r\n')
            if not line:
                continue
            entry = Entry(file_path, line_num, line)
            if entry.err_msg:
                bad_entries.append(entry)
            else:
                good_entries.append(entry)
    print_errors(bad_entries)
    return good_entries


def _check_dup(entries: List[Entry]):
    """
    중복된 엔트리가 없는 지 확인한다.
    Args:
        entries:  엔트리 리스트
    """
    bad_entries = []
    key_dic = {}
    for entry in entries:
        if entry.word in key_dic:
            dup_entry = key_dic[entry.word]
            entry.err_msg = '[DUPLICATED] with "{}"'.format(dup_entry)
            bad_entries.append(entry)
        else:
            key_dic[entry.word] = entry
    print_errors(bad_entries)


def _set_align(aligner: Aligner, Word: type, entries: List[Entry]):    # pylint: disable=invalid-name
    """
    음절과 형태소 분석 결과를 정렬한다.
    Args:
        aligner:  Aligner 객체
        Word:  Word 타입
        entries:  엔트리 리스트
    """
    bad_entries = []
    for entry in entries:
        if entry.is_sharp:
            continue
        word = Word.parse('\t'.join(['', entry.word, Morph.to_str(entry.morphs)]), '', 0)
        try:
            entry.align = aligner.align(word)
        except AlignError as map_exc:
            entry.err_msg = 'fail to align'
            logging.error(map_exc)
            bad_entries.append(entry)
    print_errors(bad_entries)


def _set_tag_out(restore_dic: dict, restore_new: dict, vocab_out: Dict[str, int],
                 vocab_new: Dict[str, int], entries: List[Entry]):
    """
    음절 정렬로부터 출력 태그를 결정하고 출력 태그의 번호를 매핑한다.
    Args:
        restore_dic:  원형복원 사전
        restore_new:  원형복원 사전에 추가할 엔트리
        vocab_out:  출력 태그 사전
        vocab_new:  출력 태그 사전에 추가할 새로운 태그
        entries:  엔트리 리스트
    """
    for entry in entries:
        entry.tag_outs, entry.tag_nums = align_to_tag(entry.word, entry.align,
                                                      (restore_dic, restore_new),
                                                      (vocab_out, vocab_new))


def _save_trie(rsc_dir: str, entries: List[Entry]):
    """
    트라이를 저장한다.
    Args:
        rsc_dir:  대상 리소스 디렉토리
        entries:  엔트리 리스트
    """
    trie = Trie()
    total_tag_nums = 0
    for entry in entries:
        val = total_tag_nums
        val += 1    # 인덱스는 0이 아니라 1부터 시작한다.
        val *= 2    # 어절 완전일치의 경우 짝수
        val += 1 if entry.is_pfx else 0    # 전망매칭 패턴의 경우 홀수
        trie.insert(entry.word, val)
        total_tag_nums += len(entry.tag_nums)
    trie.save('{}/preanal.tri'.format(rsc_dir))

    val_file = '{}/preanal.val'.format(rsc_dir)
    with open(val_file, 'wb') as fout:
        fout.write(struct.pack('H', 0))    # 인덱스가 1부터 시작하므로 dummy 데이터를 맨 앞에 하나 넣는다.
        for idx, entry in enumerate(entries, start=1):
            logging.debug('%d: %s: %s: %s', idx, entry.word, entry.tag_outs, entry.tag_nums)
            fout.write(struct.pack('H' * len(entry.tag_nums), *entry.tag_nums))
    logging.info('value saved: %s', val_file)
    logging.info('total entries: %d', len(entries))
    logging.info('expected size: %d',
                 (sum([len(e.tag_nums) for e in entries])+1) * struct.Struct('H').size)


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    aligner = Aligner(args.rsc_src)
    restore_dic = load_restore_dic('{}/restore.dic'.format(args.rsc_src))
    if not restore_dic:
        sys.exit(1)
    restore_new = defaultdict(dict)
    vocab_out = load_vocab_out(args.rsc_src)
    vocab_new = {}

    entries = _load_entries(args)
    _check_dup(entries)
    entries = [e for e in entries if not e.is_sharp]    # 주석 처리한 엔트리는 제외
    _set_align(aligner, sejong_corpus.Word, entries)
    _set_tag_out(restore_dic, restore_new, vocab_out, vocab_new, entries)

    append_new_entries(args.rsc_src, restore_new, vocab_new)
    _save_trie(args.rsc_dir, entries)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='기분석 사전을 빌드하는 스크립트')
    parser.add_argument('--rsc-src', help='source directory (text) <default: ./src>',
                        metavar='DIR', default='./src')
    parser.add_argument('--rsc-dir', help='target directory (binary) <default: ./share/khaiii>',
                        metavar='DIR', default='./share/khaiii')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
