#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
compile error patch
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
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


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


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
            file_path:  file path
            line_num:  line number
            line:  raw line
        """
        self.file_name = os.path.basename(file_path)
        self.line_num = line_num
        self.line = line
        self.err_msg = ''    # error message if there exists an error
        self.is_sharp = False    # whether does start with sharp (comment line)
        self.raw = ''    # raw patch string
        self.left = []    # left side of patch (error results)
        self.left_align = []    # alignment of left side
        self.right = []    # right side of patch (correct results)
        self.right_align = []    # alignment of right side
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
        make key string for duplication check using raw string and left side of patch
        Returns:
            the key string
        """
        return '{}\t{}'.format(self.raw, Morph.to_str(self.left))

    def _parse(self):
        """
        parse a single patch line
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
    load patch entries from file
    Args:
        args:  program arguments
    Returns:
        list of entries
    """
    good_entries = []
    bad_entries = []
    for file_path in glob.glob('{}/errpatch.*'.format(args.rsc_src)):
        file_name = os.path.basename(file_path)
        _LOG.info(file_name)
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
    check if there exist duplicated entries
    Args:
        entries:  list of entries
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
    align characters with analyzed results
    Args:
        rsc_src:  (Aligner, restore dic, vocab out) resource triple
        entries:  list of entries
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
    save trie
    Args:
        rsc_dir:  target resource directory
        entries:  list of entries
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
        _LOG.debug('%s:%s => %s => %d => %s', entry.raw, entry.left_align, key, val,
                   entry.right_align)
        rights.append(entry.right_align)
        total_patch += 1
    trie.save('{}/errpatch.tri'.format(rsc_dir))

    len_file = '{}/errpatch.len'.format(rsc_dir)
    with open(len_file, 'wb') as fout:
        # since index starts with 1, insert dummy data at the first
        fout.write(struct.pack('B', 0))
        for idx, right in enumerate(rights, start=1):
            right.append(0)
            fout.write(struct.pack('B', len(right)))
    _LOG.info('length saved: %s', len_file)
    _LOG.info('expected size: %d', len(rights)+1)

    val_file = '{}/errpatch.val'.format(rsc_dir)
    with open(val_file, 'wb') as fout:
        # since index starts with 1, insert dummy data at the first
        fout.write(struct.pack('h', 0))
        for idx, right in enumerate(rights, start=1):
            _LOG.debug('%d: %s (%d)', idx, right, len(right))
            right.append(0)
            fout.write(struct.pack('h' * len(right), *right))
    _LOG.info('value saved: %s', val_file)
    _LOG.info('total entries: %d', len(rights))
    _LOG.info('expected size: %d', (sum([len(r) for r in rights])+1) * struct.Struct('h').size)


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
        _LOG.error('no entry to compile')
        sys.exit(2)
    _check_dup(entries)
    entries = [e for e in entries if not e.is_sharp]    # exclude sharped(commented) entries
    _set_align((aligner, restore_dic, vocab_out), entries)
    _save_trie(args.rsc_dir, entries)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='compile error patch')
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
