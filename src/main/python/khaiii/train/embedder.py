# -*- coding: utf-8 -*-


"""
making embedding models
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import logging
import math

import torch
from torch import nn, Tensor

from khaiii.train.dataset import FIELDS


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


#########
# types #
#########
class Embedder(nn.Module):
    """
    embedder class
    """
    def __init__(self, cfg: Namespace, padding_idx: int = 0):
        """
        Args:
            cfg:  config
            padding_idx:  padding index
        """
        super().__init__()
        self.cfg = cfg
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(cfg.vocab_in, cfg.embed_dim, padding_idx=padding_idx)
        self.zeros = None    # zero embedding to check padding (for debug)
        self.spc_dropout_mod = nn.Dropout(p=cfg.spc_dropout)

    def _spc_dropout(self, tensor: Tensor) -> Tensor:
        """
        apply space drop out to the tensor
        Args:
            tensor:  tensor to apply drop out
        Returns:
            drop-outed tensor
        """
        if self.cfg.spc_dropout <= 0 or not self.training:
            return tensor
        return self.spc_dropout_mod(tensor.type(torch.float32)).type(torch.long)    # pylint: disable=no-member

    def _check_padding(self):
        """
        check padding embedding whether its values of vector are zero
        """
        # padding embedding must be zero vector, but it changes by backward step
        # force overwriting padding embedding with zeros
        if self.zeros is None:
            device = torch.device(f'cuda:{self.cfg.gpu_num}' if self.cfg.gpu_num >= 0 else 'cpu')    # pylint: disable=no-member
            self.zeros = torch.zeros([1, self.cfg.embed_dim], device=device)    # pylint: disable=no-member
        if not (self.embedding.weight[self.padding_idx] == self.zeros).all():
            _LOG.debug('self.embedding.weight[%d]:\n%s', self.padding_idx,
                       self.embedding.weight[self.padding_idx])
            with torch.no_grad():
                self.embedding.weight[self.padding_idx] = 0
            _LOG.debug('self.embedding.weight[%s]:\n%s', self.padding_idx,
                       self.embedding.weight[self.padding_idx])
        assert (self.embedding.weight[self.padding_idx] == self.zeros).all(), \
               f'padding embedding must be zero vector:\n{self.embedding.weight[self.padding_idx]}'

    def forward(self, batch: Tensor, use_spc_mask: bool):    # pylint: disable=arguments-differ
        """
        Args:
            batch:  batch input
            use_spc_mask:  whether use left/right space masks or not
        Returns:
            embedding vectors
        """
        self._check_padding()
        _LOG.debug('batch.char:\n%s', batch.char)
        embeds = self.embedding(batch.char)
        _LOG.debug('embeds:\n%s', embeds)
        if use_spc_mask:
            left_spc = self._spc_dropout(batch.left_spc)
            right_spc = self._spc_dropout(batch.right_spc)
            embeds += self.embedding(left_spc * FIELDS['char'].vocab.stoi['<w>'])
            embeds += self.embedding(right_spc * FIELDS['char'].vocab.stoi['</w>'])
        # 왼쪽과 오른쪽 패딩에는 zero 벡터인데 아래 positional encoding이 더해짐
        # 사소하지만 아래도 패딩 영역에 대해 마스킹 후 더해줘야 하지 않을까?
        embeds += positional_encoding(self.cfg.context_len, self.cfg.context_len,
                                      self.cfg.embed_dim, 1, self.cfg.gpu_num)
        return embeds


#############
# functions #
#############
def memoize(func):
    """
    memoize decorator
    """
    class Memodict(dict):
        """
        Memoization decorator for a function taking one or more arguments.
        """
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = func(*key)
            return ret

    return Memodict().__getitem__


@memoize
def positional_encoding(sent_len: int, max_dim: int, embed_dim: int, method: int = 1,
                        gpu_num: int = -1) -> Tensor:
    """
    positional encoding Tensor 출력.
    embeds [batch_size, context_len, embed_dim]에 Broadcasting 으로 더해짐
    Args:
        sent_len:  actual sentence length
        max_dim:  maximum dimension
        embed_dim:  embedding dimension
        method:  method number (1. end-to-end memory networks or 2. attention is all you need)
        gpu_num:  GPU device number. default: -1 for CPU
    Returns:
        pe [context_len, embed_dim]
    """
    device = gpu_num if gpu_num >= 0 else None
    pe_tensor = torch.zeros([max_dim, embed_dim], device=device)    # pylint: disable=no-member
    for pos in range(1, sent_len + 1):
        for i in range(1, embed_dim+1):
            if method == 1:
                # end-to-end memory networks
                pe_tensor[pos-1, i-1] = 1 - pos / sent_len - ((i / embed_dim) *
                                                              (1 - 2 * pos / sent_len))
            elif method == 2:
                # attention is all you need
                if i % 2 == 0:
                    pe_tensor[pos-1, i-1] = math.sin(pos / 10000 ** (2*i / embed_dim))
                else:
                    pe_tensor[pos-1, i-1] = math.cos(pos / 10000 ** (2*i / embed_dim))
    pe_tensor.detach()
    return pe_tensor
