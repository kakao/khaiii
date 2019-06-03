#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
오분석 패치를 빌드하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import glob
import logging
import os
import struct
import sys
from typing import Dict, List, Tuple

from khaiii.resource.char_align import Aligner, align_patch
from khaiii.resource.resource import load_vocab_out, parse_restore_dic
from khaiii.resource.morphs import Morph, ParseError, mix_char_tag
from khaiii.resource.trie import Trie

from compile_preanal import print_errors


#########
# types #
#########
class Entry:
    """
    error patch entry
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

    def key_str(self) -> str:
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
def _load_entries(args: Namespace) -> List[Entry]:
    """
    패치 엔트리를 파일로부터 로드한다.
    Args:
        args:  program arguments
    Returns:
        엔트리 리스트
    """
    good_entries = []
    bad_entries = []
    for file_path in glob.glob('{}/{}.errpatch.*'.format(args.rsc_src, args.model_size)):
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
        if entry.key_str() in key_dic:
            dup_entry = key_dic[entry.key_str()]
            entry.err_msg = '[DUPLICATED] with "{}"'.format(dup_entry)
            bad_entries.append(entry)
        else:
            key_dic[entry.key_str()] = entry
    print_errors(bad_entries)


def _set_align(rsc_src: Tuple[Aligner, dict, Dict[str, int]], entries: List[Entry]):
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


def _save_trie(rsc_dir: str, entries: List[Entry]):
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
    trie.save('{}/errpatch.tri'.format(rsc_dir))

    len_file = '{}/errpatch.len'.format(rsc_dir)
    with open(len_file, 'wb') as fout:
        fout.write(struct.pack('B', 0))    # 인덱스가 1부터 시작하므로 dummy 데이터를 맨 앞에 하나 넣는다.
        for idx, right in enumerate(rights, start=1):
            right.append(0)
            fout.write(struct.pack('B', len(right)))
    logging.info('length saved: %s', len_file)
    logging.info('expected size: %d', len(rights)+1)

    val_file = '{}/errpatch.val'.format(rsc_dir)
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


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    aligner = Aligner(args.rsc_src)
    restore_dic = parse_restore_dic('{}/restore.dic'.format(args.rsc_src))
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
    parser = ArgumentParser(description='기분석 사전을 빌드하는 스크립트')
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
