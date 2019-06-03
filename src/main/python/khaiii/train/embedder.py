# -*- coding: utf-8 -*-


"""
making embedding models
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import math

import torch
from torch import nn, Tensor

from khaiii.resource.resource import Resource


class Embedder(nn.Module):
    """
    embedder class
    """
    def __init__(self, cfg: Namespace, rsc: Resource):
        """
        Args:
            cfg:  config
            rsc:  Resource object
        """
        super().__init__()
        self.cfg = cfg
        self.rsc = rsc
        self.embedding = nn.Embedding(len(rsc.vocab_in), cfg.embed_dim, 0)

    def forward(self, *inputs):    # pylint: disable=arguments-differ
        """
        임베딩을 생성하는 메소드
        Args:
            inputs:  batch size list of (context, left space mask, right space mask)
        Returns:
            embedding
        """
        contexts, left_spc_masks, right_spc_masks = inputs
        embeds = self.embedding(contexts)
        if left_spc_masks is not None:
            embeds += self.embedding(left_spc_masks)
        if right_spc_masks is not None:
            embeds += self.embedding(right_spc_masks)
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
