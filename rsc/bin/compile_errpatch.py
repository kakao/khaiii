#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
오분석 패치를 빌드하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import argparse
from collections import defaultdict
import glob
import itertools
import logging
import os
import struct
import sys

from char_align import Aligner, AlignError
from compile_preanal import align_to_tag, print_errors
from compile_restore import load_restore_dic, load_vocab_out
from morphs import Morph, ParseError
from morphs import WORD_DELIM_STR, SENT_DELIM_STR, WORD_DELIM_NUM, SENT_DELIM_NUM
import sejong_corpus
from trie import Trie


#########
# types #
#########
class Entry:
    """
    error patch entry
    """
    def __init__(self, file_path, line_num, line):
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
        self.raw = ''    # 패치 원문
        self.left = []    # 패치 좌측 (오분석)
        self.left_align = []    # 패치 좌측 정렬 정보
        self.right = []    # 패치 우측 (정분석)
        self.right_align = []    # 패치 우측 정렬 정보
        self._parse()

    def __str__(self):
        file_num = '{}:{}: '.format(self.file_name, self.line_num) if self.file_name else ''
        line = '# {}'.format(self.line) if self.is_sharp else self.line
        if self.err_msg:
            return '{}{}: "{}"'.format(file_num, self.err_msg, line)
        if self.is_sharp:
            return '{}: "{}"'.format(file_num, line)
        return '{}\t{}\t{}'.format(self.raw, Morph.to_str(self.left), Morph.to_str(self.right))

    def key_str(self):
        """
        패치의 중복 검사를 하기 위해 원문과 left를 이용하여 키를 생성
        Returns:
            중복 생성 검사를 위한 키
        """
        return '{}\t{}'.format(self.raw, Morph.to_str(self.left))

    def _parse(self):
        """
        오분석 패치 한 라인을 파싱한다.
        """
        if len(self.line) >= 2 and self.line.startswith('# '):
            self.is_sharp = True
            self.line = self.line[2:]
        cols = self.line.split('\t')
        if len(cols) != 3:
            if not self.is_sharp:
                self.err_msg = '[PARSE] number of columns must be 3, not {}'.format(len(cols))
            return
        self.raw, left_str, right_str = cols
        if not self.raw:
            self.err_msg = '[PARSE] no raw string'
            return
        try:
            self.left = Morph.parse(left_str)
        except ParseError as par_err:
            self.err_msg = '[PARSE] {}'.format(par_err)
        try:
            self.right = Morph.parse(right_str)
        except ParseError as par_err:
            self.err_msg = '[PARSE] {}'.format(par_err)


#############
# functions #
#############
def _split_list(lst, delim):
    """
    리스트를 delimiter로 split하는 함수

    >>> _split_list(['가/JKS', '_', '너/NP'], '_')
    [['가/JKS'], ['너/NP']]

    Args:
        lst:  리스트
        delim:  delimiter
    Returns:
        list of sublists
    """
    sublists = []
    while lst:
        prefix = [x for x in itertools.takewhile(lambda x: x != delim, lst)]
        sublists.append(prefix)
        lst = lst[len(prefix):]
        delims = [x for x in itertools.takewhile(lambda x: x == delim, lst)]
        lst = lst[len(delims):]
    return sublists


def align_patch(rsc_src, raw, morph_str):
    """
    패치의 원문과 분석 결과를 음절단위 매핑(정렬)을 수행한다.
    Args:
        rsc_src:  (Aligner, restore dic, vocab out) resource triple
        raw:  원문
        morph_str:  형태소 분석 결과 (패치 기술 형식)
    Returns:
        정렬에 기반한 출력 태그 번호
    """
    aligner, restore_dic, vocab_out = rsc_src
    raw_words = raw.strip().split()
    morphs = morph_str.split(' + ')
    morphs_strip = morphs
    if morphs[0] in [WORD_DELIM_STR, SENT_DELIM_STR]:
        morphs_strip = morphs_strip[1:]
    if morphs[-1] in [WORD_DELIM_STR, SENT_DELIM_STR]:
        morphs_strip = morphs_strip[:-1]
    morph_words = _split_list(morphs_strip, WORD_DELIM_STR)
    tag_nums = []
    restore_new = defaultdict(dict)
    vocab_new = defaultdict(list)
    for raw_word, morph_word in zip(raw_words, morph_words):
        word = sejong_corpus.Word.parse('\t'.join(['', raw_word, ' + '.join(morph_word)]), '', 0)
        try:
            word_align = aligner.align(word)
            _, word_tag_nums = \
                align_to_tag(raw_word, word_align, (restore_dic, restore_new),
                             (vocab_out, vocab_new))
            if restore_new or vocab_new:
                logging.debug('needs dic update: %s', word)
                return []
        except AlignError as algn_err:
            logging.debug('alignment error: %s', word)
            logging.debug(str(algn_err))
            return []
        if tag_nums:
            tag_nums.append(WORD_DELIM_NUM)
        tag_nums.extend(word_tag_nums)
    if morphs[0] in [WORD_DELIM_STR, SENT_DELIM_STR]:
        tag_nums.insert(0, WORD_DELIM_NUM if morphs[0] == WORD_DELIM_STR else SENT_DELIM_NUM)
    if morphs[-1] in [WORD_DELIM_STR, SENT_DELIM_STR]:
        tag_nums.append(WORD_DELIM_NUM if morphs[-1] == WORD_DELIM_STR else SENT_DELIM_NUM)
    return tag_nums


def mix_char_tag(chars, tags):
    """
    음절과 출력 태그를 비트 연산으로 합쳐서 하나의 (32비트) 숫자로 표현한다.
    Args:
        chars:  음절 (유니코드) 리스트 (문자열)
        tags:  출력 태그 번호의 리스트
    Returns:
        합쳐진 숫자의 리스트
    """
    char_nums = [ord(c) for c in chars]
    if tags[0] == SENT_DELIM_NUM:
        char_nums.insert(0, SENT_DELIM_NUM)
    if tags[-1] == SENT_DELIM_NUM:
        char_nums.append(SENT_DELIM_NUM)
    for idx, char_num in enumerate(char_nums):
        if char_num == ord(' '):
            char_nums[idx] = WORD_DELIM_NUM
            continue
        elif tags[idx] == SENT_DELIM_NUM:
            continue
        char_nums[idx] = char_num << 12 | tags[idx]
    return char_nums


def _load_entries(args):
    """
    패치 엔트리를 파일로부터 로드한다.
    Args:
        args:  arguments
    Returns:
        엔트리 리스트
    """
    good_entries = []
    bad_entries = []
    for file_path in glob.glob(f'{args.rsc_src}/{args.model_size}.errpatch.*'):
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


def _check_dup(entries):
    """
    중복된 엔트리가 없는 지 확인한다.
    Args:
        entries:  엔트리 리스트
    """
    bad_entries = []
    key_dic = {}
    for entry in entries:
        if entry.key_str() in key_dic:
            dup_entry = key_dic[entry.key_str()]
            entry.err_msg = '[DUPLICATED] with "{}"'.format(dup_entry)
            bad_entries.append(entry)
        else:
            key_dic[entry.key_str()] = entry
    print_errors(bad_entries)


def _set_align(rsc_src, entries):    # pylint: disable=invalid-name
    """
    음절과 형태소 분석 결과를 정렬한다.
    Args:
        rsc_src:  (Aligner, restore dic, vocab out) resource triple
        Word:  Word 타입
        entries:  엔트리 리스트
    """
    bad_entries = []
    for entry in entries:
        if entry.is_sharp:
            continue
        entry.left_align = align_patch(rsc_src, entry.raw, Morph.to_str(entry.left))
        if not entry.left_align:
            entry.err_msg = 'fail to align left'
            bad_entries.append(entry)
            continue
        entry.right_align = align_patch(rsc_src, entry.raw, Morph.to_str(entry.right))
        if not entry.right_align:
            entry.err_msg = 'fail to align right'
            bad_entries.append(entry)
            continue
        assert len(entry.left_align) == len(entry.right_align)
    print_errors(bad_entries)


def _save_trie(rsc_dir, entries):
    """
    트라이를 저장한다.
    Args:
        rsc_dir:  대상 리소스 디렉토리
        entries:  엔트리 리스트
    """
    trie = Trie()
    total_patch = 0
    rights = []
    for entry in entries:
        if entry.is_sharp:
            continue
        val = total_patch + 1
        key = mix_char_tag(entry.raw, entry.left_align)
        trie.insert(key, val)
        logging.debug('%s:%s => %s => %d => %s', entry.raw, entry.left_align, key, val,
                      entry.right_align)
        rights.append(entry.right_align)
        total_patch += 1
    trie.save(f'{rsc_dir}/errpatch.tri')

    len_file = f'{rsc_dir}/errpatch.len'
    with open(len_file, 'wb') as fout:
        fout.write(struct.pack('B', 0))    # 인덱스가 1부터 시작하므로 dummy 데이터를 맨 앞에 하나 넣는다.
        for idx, right in enumerate(rights, start=1):
            right.append(0)
            fout.write(struct.pack('B', len(right)))
    logging.info('length saved: %s', len_file)
    logging.info('expected size: %d', len(rights)+1)

    val_file = f'{rsc_dir}/errpatch.val'
    with open(val_file, 'wb') as fout:
        fout.write(struct.pack('h', 0))    # 인덱스가 1부터 시작하므로 dummy 데이터를 맨 앞에 하나 넣는다.
        for idx, right in enumerate(rights, start=1):
            logging.debug('%d: %s (%d)', idx, right, len(right))
            right.append(0)
            fout.write(struct.pack('h' * len(right), *right))
    logging.info('value saved: %s', val_file)
    logging.info('total entries: %d', len(rights))
    logging.info('expected size: %d',
                 (sum([len(r) for r in rights])+1) * struct.Struct('h').size)


def run(args):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    aligner = Aligner(args.rsc_src)
    restore_dic = load_restore_dic(f'{args.rsc_src}/restore.dic')
    if not restore_dic:
        sys.exit(1)
    vocab_out = load_vocab_out(args.rsc_src)

    entries = _load_entries(args)
    if not entries:
        logging.error('no entry to compile')
        sys.exit(2)
    _check_dup(entries)
    entries = [e for e in entries if not e.is_sharp]    # 주석 처리한 엔트리는 제외
    _set_align((aligner, restore_dic, vocab_out), entries)
    _save_trie(args.rsc_dir, entries)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='기분석 사전을 빌드하는 스크립트')
    parser.add_argument('--model-size', help='model size <default: base>',
                        metavar='SIZE', default='base')
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
