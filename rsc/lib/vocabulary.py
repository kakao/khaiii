# -*- coding: utf-8 -*-


"""
vocabulary library
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import re
import codecs
from collections import defaultdict
import copy
import logging
import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


#########
# types #
#########
class Vocabulary(object):
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


class PreTrainedVocabulary(Vocabulary):
    """
    pre-train된 word2vec를 사용하는 경우, vector에 있는 어휘로
    사전을 구성하도록 합니다.
    """
    def __init__(self, path): #pylint: disable=super-init-not-called
        """
        Args:
            path: file path
        """
        # simple : 사과/N , none : 사과
        # 읽어들인 glove의 키 타입을 보고 판단해놓는다.
        self.glove_key_type = None
        self.dic, self.vectors = self._load_glove(path)
        self.rev = {val:key for key, val in self.dic.items()}
        assert len(self.dic) == len(self.rev)
        logging.info('%s: %d entries, %d dim - not trainable',
                     os.path.basename(path), len(self.dic), self.vectors.size(1))

    def get_embedding(self, dim, padding_idx=None):
        """
        pre-training된 벡터가 세팅된 embedding을 리턴합니다.
        """
        assert dim == self.vectors.size(1)
        embed = super().get_embedding(dim, padding_idx)
        embed.weight = nn.Parameter(self.vectors, requires_grad=False)
        return embed

    def _load_glove(self, path):
        """
        pre-trained GloVe (텍스트 포맷) 워드 벡터를 읽어들인다.
        Args:
            path:  워드 벡터 경로
        """
        unk = None
        vecs = []
        for line in tqdm(codecs.open(path, 'r', encoding='UTF-8')):
            cols = line.split(' ')
            word = cols[0]
            vec = np.array([float(_) for _ in cols[1:]])
            if vec.size == 0: # format error
                continue
            if word == '<unk>':
                unk = vec
                continue
            vecs.append((word, vec))
            if self.glove_key_type is None:
                if re.search('/[A-Z]$', word) is None:
                    self.glove_key_type = 'none'
                else:
                    self.glove_key_type = 'simple'
        if unk is None:
            unk = [0] * len(vecs[0][1])
        padding = [0] * len(vecs[0][1])
        vecs.sort(key=lambda x: x[0])
        vecs.insert(0, ('<unk>', unk))
        vecs.insert(1, ('<p>', padding))
        vocab = defaultdict(int)
        vocab.update({word: idx for idx, (word, _) in enumerate(vecs)})
        return vocab, torch.Tensor([vec for _, vec in vecs])
