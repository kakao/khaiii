# -*- coding: utf-8 -*-


"""
resources for training and tagging
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
from collections import defaultdict
import logging
import os
from typing import Dict, Tuple

from khaiii.resource.vocabulary import VocabIn, VocabOut


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


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
        self.vocab_in = VocabIn(vocab_in_path, cfg.min_freq)
        vocab_out_path = '{}/vocab.out'.format(cfg.rsc_src)
        self.vocab_out = VocabOut(vocab_out_path)
        restore_dic_path = '{}/restore.dic'.format(cfg.rsc_src)
        self.restore_dic = self.load_restore_dic(restore_dic_path)

    @classmethod
    def load_restore_dic(cls, path: str) -> Dict[str, str]:
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
        _LOG.info('%s: %d entries', os.path.basename(path), len(dic))
        return dic


#############
# functions #
#############
def parse_restore_dic(file_path: str) -> Dict[Tuple[str, str], Dict[int, str]]:
    """
    load the restore dictionary
    Args:
        file_path:  file path
    Returns:
        the dictionary
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
                _LOG.error('%s:%d: duplicated with %s: %s', file_name, line_num,
                           num_mrp_chrs_dic[num], line)
                return {}
        restore_dic[char, tag][num] = mrp_chr_str
    return restore_dic


def load_vocab_out(rsc_src: str) -> Dict[str, int]:
    """
    load the output vocabulary
    Args:
        rsc_src:  resource directory
    Returns:
        the vocabulary
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
