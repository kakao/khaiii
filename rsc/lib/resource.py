# -*- coding: utf-8 -*-


"""
resources for training and tagging
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import codecs
import logging
import os

from vocabulary import Vocabulary


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
class Resource(object):
    """
    resources
    """
    def __init__(self, cfg):
        """
        :param  cfg:  config
        """
        vocab_in_path = '{}/vocab.in'.format(cfg.rsc_src)
        self.vocab_in = Vocabulary(vocab_in_path, cfg.cutoff, SPECIAL_CHARS)
        vocab_out_path = '{}/vocab.out'.format(cfg.rsc_src)
        self.vocab_out = Vocabulary(vocab_out_path, 0, None)
        restore_dic_path = '{}/restore.dic'.format(cfg.rsc_src)
        self.restore_dic = self._load_restore_dic(restore_dic_path)

    @classmethod
    def _load_restore_dic(cls, path):
        """
        load character to output tag mapping
        :param  path:  file path
        :return:  dictionary
        """
        dic = {}
        for line in codecs.open(path, 'r', encoding='UTF-8'):
            line = line.rstrip('\r\n')
            if not line:
                continue
            key, val = line.split('\t')
            dic[key] = val
        logging.info('%s: %d entries', os.path.basename(path), len(dic))
        return dic
