# -*- coding: utf-8 -*-


"""
Pytorch models
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from khaiii.resource.resource import Resource
from khaiii.train.embedder import Embedder


#########
# types #
#########
class ConvLayer(nn.Module):
    """
    형태소 태깅 모델과 띄어쓰기 모델이 공유하는 컨볼루션 레이어
    """
    def __init__(self, cfg: Namespace, rsc: Resource):
        """
        Args:
            cfg:  config
            rsc:  Resource object
        """
        super().__init__()
        self.embedder = Embedder(cfg, rsc)
        ngram = min(5, cfg.window * 2 + 1)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size)
                                    for kernel_size in range(2, ngram+1)])

    def forward(self, *inputs):
        embeds = self.embedder(*inputs)
        embeds_t = embeds.transpose(1, 2)
        pool_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embeds_t))
            pool_outs.append(F.max_pool1d(conv_out, conv_out.size(2)))
        features = torch.cat([p.view(embeds.size(0), -1) for p in pool_outs], dim=1)    # pylint: disable=no-member
        return features


class Model(nn.Module):
    """
    형태소 태깅 모델, 띄어쓰기 모델
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
        self.conv_layer = ConvLayer(cfg, rsc)
        hidden_dim = (cfg.embed_dim * len(self.conv_layer.convs) + len(rsc.vocab_out)) // 2
        setattr(cfg, 'hidden_dim', hidden_dim)
        self.hidden_layer_pos = nn.Sequential(OrderedDict([
            ('feature_dropout', nn.Dropout(p=cfg.hdn_dropout)),
            ('conv2hidden', nn.Linear(cfg.embed_dim * len(self.conv_layer.convs), hidden_dim)),
            ('relu1', nn.ReLU()),
            ('hidden_dropout', nn.Dropout(p=cfg.hdn_dropout)),
            ('hidden2logit', nn.Linear(hidden_dim, len(rsc.vocab_out))),
        ]))
        self.hidden_layer_spc = nn.Sequential(
            nn.Dropout(p=cfg.hdn_dropout),
            nn.Linear(cfg.embed_dim * len(self.conv_layer.convs), hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.hdn_dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, *inputs):
        contexts, left_spc_masks, right_spc_masks = inputs
        features_pos = self.conv_layer(contexts, left_spc_masks, right_spc_masks)
        features_spc = self.conv_layer(contexts, None, None)
        logits_pos = self.hidden_layer_pos(features_pos)
        logits_spc = self.hidden_layer_spc(features_spc)
        return logits_pos, logits_spc

    def save(self, path: str):
        """
        모델을 저장하는 메소드
        Args:
            path:  경로
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        저장된 모델을 로드하는 메소드
        Args:
            path:  경로
            conv_layer:  convolution layer
        """
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        if torch.cuda.is_available() and self.cfg.gpu_num >= 0:
            self.cuda(device=self.cfg.gpu_num)
