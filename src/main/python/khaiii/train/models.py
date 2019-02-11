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

import torch
import torch.nn as nn
import torch.nn.functional as F

from khaiii.resource.resource import Resource
from khaiii.train.embedder import Embedder


#########
# types #
#########
class PosModel(nn.Module):
    """
    part-of-speech tagger pytorch model
    """
    def __init__(self, cfg: Namespace, rsc: Resource):
        """
        Args:
            cfg (Namespace):  config
            rsc (Resource):  Resource object
        """
        super().__init__()
        self.cfg = cfg
        self.rsc = rsc
        self.embedder = Embedder(cfg, rsc)

    def forward(self, *inputs):
        raise NotImplementedError

    def save(self, path: str):
        """
        모델을 저장하는 메소드
        Args:
            path (str):  경로
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        저장된 모델을 로드하는 메소드
        Args:
            path (str):  경로
        """
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.cuda()


class CnnModel(PosModel):
    """
    convolutional neural network based part-of-speech tagger
    """
    def __init__(self, cfg: Namespace, rsc: Resource):
        """
        Args:
            cfg (Namespace):  config
            rsc (Resource):  Resource object
        """
        super().__init__(cfg, rsc)

        ngram = min(5, cfg.window * 2 + 1)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size)
                                    for kernel_size in range(2, ngram+1)])

        # conv => hidden
        setattr(cfg, 'hidden_dim', (cfg.embed_dim * len(self.convs) + len(rsc.vocab_out)) // 2)
        self.conv2hidden = nn.Linear(cfg.embed_dim * len(self.convs), cfg.hidden_dim)

        # hidden => tag
        self.hidden2tag = nn.Linear(cfg.hidden_dim, len(rsc.vocab_out))

    def forward(self, inputs):    # pylint: disable=arguments-differ
        """
        forward path
        Args:
            inputs:  batch size list of (context, left space mask, right space mask)
        Returns:
            output score
        """
        embeds = self.embedder(inputs)
        embeds_t = embeds.transpose(1, 2)

        pool_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embeds_t))
            pool_outs.append(F.max_pool1d(conv_out, conv_out.size(2)))

        # conv => hidden
        features = torch.cat([p.view(embeds.size(0), -1) for p in pool_outs], dim=1)    # pylint: disable=no-member
        features_drop = F.dropout(features)
        hidden_out = F.relu(self.conv2hidden(features_drop))

        # hidden => tag
        hidden_out_drop = F.dropout(hidden_out)
        tag_out = self.hidden2tag(hidden_out_drop)
        return tag_out
