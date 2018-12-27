# -*- coding: utf-8 -*-


"""
vocabulary library
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import codecs
import copy
import logging
import os


#########
# types #
#########
class Vocabulary:
    """
    vocabulary class
    """
    def __init__(self, path, cutoff=1, special=None, padding=''):
        """
        Args:
            path:  file path
            cutoff:  cutoff frequency
            special:  special entries located at the first
            padding:  add padding special char at the end
        """
        self.dic = {}    # {entry: number} dictionary
        self.rev = copy.deepcopy(special) if special else []    # reverse dictionary
        for num, entry in enumerate(self.rev):
            self.dic[entry] = num
        self._load(path, cutoff)
        self.padding = padding
        if padding:
            if padding in self.dic:
                raise ValueError('padding special character already in vocab: {}'.format(padding))
            padding_idx = len(self.dic)
            self.dic[padding] = padding_idx
            self.rev.append(padding)
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
        except KeyError:
            return 0    # unknown word number

    def __len__(self):
        return len(self.dic)

    '''
    # 리소스 빌드 시 pytorch 의존성 제거를 위해 임시로 메서드를 제거합니다.
    # 추후 학습 코드를 추가할 때 이 부분을 리팩토링 합니다.
    def get_embedding(self, dim, padding_idx=None):
        """
        embedding을 리턴합니다.
        Args:
            dim:  embedding dimension
            padding_idx:  padding index
        """
        if padding_idx:
            return nn.Embedding(len(self), dim, padding_idx=padding_idx)
        return nn.Embedding(len(self), dim)
    '''    # pylint: disable=pointless-string-statement

    def padding_idx(self):
        """
        맨 마지막에 추가한 패딩의 인덱스를 리턴한다.
        Returns:
            패딩 인덱스
        """
        if not self.padding:
            raise RuntimeError('vocabulary has no padding')
        return self.dic[self.padding]

    def _load(self, path, cutoff=1):
        """
        load vocabulary from file
        Args:
            path:  file path
            cutoff:  cutoff frequency
        """
        append_num = 0
        cutoff_num = 0
        for line in codecs.open(path, 'r', encoding='UTF-8'):
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
