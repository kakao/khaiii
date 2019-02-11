# -*- coding: utf-8 -*-


"""
vocabulary library
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import logging
import os
from typing import List


#########
# types #
#########
class Vocabulary:
    """
    vocabulary class
    """
    def __init__(self, path: str, cutoff: int = 1, unk: str = '', special: List[str] = None):
        """
        padding index is always 0. None and '' get padding index.
        if `unk` is given (such as input vocab), its index is always 1.
        if `unk` is not given (such as output vocab), an exception will be thrown for unknown entry
        Args:
            path:  file path
            cutoff:  cutoff frequency
            unk:  unknown(OOV) entry
            special:  special entries located at the first
        """
        self.dic = {}    # {entry: number} dictionary
        self.unk = unk
        self.rev = ['', unk] if unk else ['', ]    # reverse dictionary
        if special:
            self.rev.extend(special)
        for num, entry in enumerate(self.rev):
            self.dic[entry] = num
        self._load(path, cutoff)
        assert len(self.dic) == len(self.rev)

    def __getitem__(self, key):
        """
        Args:
            key:  key
        Returns:
            word number for string key, word for int key
        """
        if isinstance(key, int):
            return self.rev[key]
        try:
            return self.dic[key]
        except KeyError as key_err:
            if self.unk:
                return self.dic[self.unk]
            raise key_err

    def __len__(self):
        return len(self.dic)

    def _load(self, path: str, cutoff: int = 1):
        """
        load vocabulary from file
        Args:
            path:  file path
            cutoff:  cutoff frequency
        """
        append_num = 0
        cutoff_num = 0
        for line in open(path, 'r', encoding='UTF-8'):
            line = line.rstrip('\r\n')
            if not line:
                continue
            try:
                entry, freq = line.split('\t')
                if int(freq) <= cutoff:
                    cutoff_num += 1
                    continue
            except ValueError:
                entry = line
            if entry in self.dic:
                cutoff_num += 1
                continue
            self.dic[entry] = len(self.dic)
            self.rev.append(entry)
            append_num += 1
        logging.info('%s: %d entries, %d cutoff', os.path.basename(path), append_num, cutoff_num)
