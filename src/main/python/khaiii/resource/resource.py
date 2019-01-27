# -*- coding: utf-8 -*-


"""
resources for training and tagging
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
from collections import defaultdict
import logging
import os
from typing import Dict, Tuple

from khaiii.resource.vocabulary import Vocabulary


#############
# constants #
#############
SPECIAL_CHARS = [
    '<u>',    # unknown character
    '<w>', '</w>',    # begin/end of word
    '<s>', '</s>'    # begin/end of sentence
]

PAD_CHR = '<p>'    # sepcial character for padding


#########
# types #
#########
class Resource:
    """
    resources
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            cfg:  config
        """
        vocab_in_path = '{}/vocab.in'.format(cfg.rsc_src)
        self.vocab_in = Vocabulary(vocab_in_path, cfg.cutoff, SPECIAL_CHARS)
        vocab_out_path = '{}/vocab.out'.format(cfg.rsc_src)
        self.vocab_out = Vocabulary(vocab_out_path, 0, None)
        restore_dic_path = '{}/restore.dic'.format(cfg.rsc_src)
        self.restore_dic = self._load_restore_dic(restore_dic_path)

    @classmethod
    def _load_restore_dic(cls, path: str) -> Dict[str, str]:
        """
        load character to output tag mapping
        Args:
            path:  file path
        Returns:
            dictionary
        """
        dic = {}
        for line in open(path, 'r', encoding='UTF-8'):
            line = line.rstrip('\r\n')
            if not line:
                continue
            key, val = line.split('\t')
            dic[key] = val
        logging.info('%s: %d entries', os.path.basename(path), len(dic))
        return dic


#############
# functions #
#############
def load_restore_dic(file_path: str) -> Dict[Tuple[str, str], Dict[int, str]]:
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


def load_vocab_out(rsc_src: str) -> Dict[str, int]:
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
