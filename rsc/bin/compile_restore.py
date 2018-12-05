#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
원형복원 사전을 빌드하는 스크립트
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from __future__ import print_function

import argparse
from collections import defaultdict
import logging
import os
import struct
import sys

from morphs import TAG_SET


#############
# constants #
#############
MAX_VAL_LEN = 4    # 원형복원 사전의 오른쪽 value 부분의 최대 길이 (고정 길이 컴파일을 위해)


#############
# functions #
#############
def load_restore_dic(file_path):
    """
    원형복원 사전을 로드한다.
    Args:
        file_path:  파일 경로
    Returns:
        사전
    """
    file_name = os.path.basename(file_path)
    restore_dic = defaultdict(dict)
    for line_num, line in enumerate(open(file_path, 'r', encoding='UTF-8'), start=1):
        line = line.rstrip()
        if not line or line[0] == '#':
            continue
        char_tag_num, mrp_chr_str = line.split('\t')
        char, tag_num = char_tag_num.rsplit('/', 1)
        tag, num = tag_num.rsplit(':', 1)
        num = int(num)
        if (char, tag) in restore_dic:
            num_mrp_chrs_dic = restore_dic[char, tag]
            if num in num_mrp_chrs_dic:
                logging.error('%s:%d: duplicated with %s: %s', file_name, line_num,
                              num_mrp_chrs_dic[num], line)
                return {}
        restore_dic[char, tag][num] = mrp_chr_str
    return restore_dic


def load_vocab_out(rsc_src):
    """
    출력 태그 vocabulary를 로드한다.
    Args:
        rsc_src:  리소스 디렉토리
    Returns:
        출력 태그 vocabulary
    """
    file_path = '{}/vocab.out'.format(rsc_src)
    vocab_out = [line.strip() for line in open(file_path, 'r', encoding='UTF-8')
                 if line.strip()]
    vocab_out_more = []
    file_path = '{}/vocab.out.more'.format(rsc_src)
    if os.path.exists(file_path):
        vocab_out_more = [line.strip() for line in open(file_path, 'r', encoding='UTF-8')
                          if line.strip()]
    return {tag: idx for idx, tag in enumerate(vocab_out + vocab_out_more, start=1)}


def append_new_entries(rsc_src, restore_new, vocab_new):
    """
    기분석 사전 빌드 중에 새로 추가가 필요한 사전 엔트리를 해당 사전에 추가한다.
    Args:
        rsc_src:  리스소 디렉토리
        restore_new:  원형복원 사전의 추가할 엔트리
        vocab_new:  출력 태그 vocabulary에 추가할 엔트리
    """
    if restore_new:
        with open('{}/restore.dic'.format(rsc_src), 'a', encoding='UTF-8') as fout:
            for (char, tag_out), tag_num_mrp_chr_dic in restore_new.items():
                for tag_num, mrp_chr in tag_num_mrp_chr_dic.items():
                    new_entry_str = '{}/{}:{}\t{}'.format(char, tag_out, tag_num, mrp_chr)
                    logging.info('[RESTORE] %s', new_entry_str)
                    print(new_entry_str, file=fout)
    if vocab_new:
        with open('{}/vocab.out.more'.format(rsc_src), 'a', encoding='UTF-8') as fout:
            new_tags = sorted([(num, tag) for tag, num in vocab_new.items()])
            for _, tag in new_tags:
                logging.info('[TAG] %s', tag)
                print(tag, file=fout)


def _make_bin(restore_dic, vocab_out, vocab_new):
    """
    두 텍스트 사전을 읽어들여 바이너리 형태의 key-value 사전을 만든다.
    Args:
        restore_dic:  원형복원 사전
        vocab_out:  출력 태그 사전
        vocab_new:  출력 태그 사전에 추가할 새로운 태그
    Retusns:
        바이너리 사전
    """
    bin_dic = {}
    for (char, tag), nums_out_dic in restore_dic.items():
        for num, out in nums_out_dic.items():
            out_tag = '{}:{}'.format(tag, num)
            logging.debug('%s/%s\t%s', char, out_tag, out)
            if out_tag not in vocab_out and out_tag not in vocab_new:
                out_num = len(vocab_out) + len(vocab_new) + 1
                logging.info('new output tag: [%d] %s', out_num, out_tag)
                vocab_new[out_tag] = out_num
            else:
                out_num = vocab_out[out_tag]
            key = (ord(char) << 12) | out_num
            if key in bin_dic:
                raise KeyError('duplicated key: 0x08x' % key)
            vals = [0, ] * MAX_VAL_LEN
            for idx, char_tag in enumerate(out.split()):
                if idx >= MAX_VAL_LEN:
                    raise ValueError('max value length exceeded: {} >= {}'.format(idx, MAX_VAL_LEN))
                char_val, tag_val = char_tag.rsplit('/', 1)
                val_mask = 0x00 if tag_val[0] == 'B' else 0x80
                tag_val = tag_val[2:]
                vals[idx] = (ord(char_val) << 8) | (TAG_SET[tag_val] | val_mask)
            bin_dic[key] = vals
            logging.debug('\t0x%08x => %s', key, ' '.join(['0x%08x' % val for val in vals]))
    return bin_dic


def _save_restore_dic(rsc_dir, bin_dic):
    """
    원형복원 바이너리 사전을 저장한다.
    Args:
        rsc_dir:  resource directory
        bin_dic:  binary dictionary
    """
    os.makedirs(rsc_dir, exist_ok=True)
    with open('{}/restore.key'.format(rsc_dir), 'wb') as fkey:
        with open('{}/restore.val'.format(rsc_dir), 'wb') as fval:
            for key, vals in sorted(bin_dic.items()):
                logging.debug('\t0x%08x => %s', key, ' '.join(['0x%08x' % val for val in vals]))
                fkey.write(struct.pack('I', key))
                fval.write(struct.pack('I' * len(vals), *vals))
    logging.info('restore.key: %d', 4 * len(bin_dic))
    logging.info('restore.val: %d', 4 * sum([len(vals) for vals in bin_dic.values()]))


def _save_restore_one(rsc_dir, vocab_out, vocab_new):
    """
    출력 태그 번호 별 원형복원을 하지 않는 비복원 사전을 저장한다.
    Args:
        rsc_dir:  resource directory
        vocab_out:  출력 태그 사전
        vocab_new:  출력 태그 사전에 추가할 새로운 태그
    :return:
    """
    idx_tags = sorted([(idx, tag) for tag, idx
                       in list(vocab_out.items()) + list(vocab_new.items())])
    os.makedirs(rsc_dir, exist_ok=True)
    with open('{}/restore.one'.format(rsc_dir), 'wb') as fone:
        fone.write(struct.pack('B', 0))   # index 0 is empty(filling) byte
        for idx, out_tag in idx_tags:
            one_tag = out_tag.split(':')[0]
            one_num = TAG_SET[one_tag[2:]]
            if one_tag[0] == 'I':
                one_num += len(TAG_SET)
            logging.debug('%d: 0x%02x [%s] %s', idx, one_num, one_tag, out_tag)
            fone.write(struct.pack('B', one_num))
    logging.info('restore.one: %d', 1 + len(idx_tags))


def run(args):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    restore_dic = load_restore_dic('{}/restore.dic'.format(args.rsc_src))
    if not restore_dic:
        sys.exit(1)
    vocab_out = load_vocab_out(args.rsc_src)
    if not vocab_out:
        sys.exit(2)
    vocab_new = {}

    bin_dic = _make_bin(restore_dic, vocab_out, vocab_new)

    _save_restore_dic(args.rsc_dir, bin_dic)
    _save_restore_one(args.rsc_dir, vocab_out, vocab_new)
    append_new_entries(args.rsc_src, None, vocab_new)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='기분석 사전을 빌드하는 스크립트')
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
