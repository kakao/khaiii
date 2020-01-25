# -*- coding: utf-8 -*-


"""
Pytorch models
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from khaiii.train.embedder import Embedder


#########
# types #
#########
class ConvLayer(nn.Module):
    """
    convolution feature layers shared with morpheme prediction and spacing
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            cfg:  config
        """
        super().__init__()
        self.embedder = Embedder(cfg)
        ngram = min(5, cfg.window * 2 + 1)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.embed_dim, cfg.embed_dim, kernel_size)
                                    for kernel_size in range(2, ngram+1)])

    def forward(self, batch: Tensor, use_spc_mask: bool):    # pylint: disable=arguments-differ
        """
        Args:
            batch:  batch input
            use_spc_mask:  whether use left/right space masks or not
        Returns:
            feature maps of convolution
        """
        embeds = self.embedder(batch, use_spc_mask)
        batch_size, sent_len, context_len, embedding_dim = embeds.size()
        embeds_r = embeds.view(batch_size * sent_len, context_len, embedding_dim)
        embeds_t = embeds_r.transpose(1, 2)
        pool_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embeds_t))
            pool_outs.append(F.max_pool1d(conv_out, conv_out.size(2)))
        features = torch.cat([p.view(embeds_r.size(0), -1) for p in pool_outs], dim=1)    # pylint: disable=no-member
        return features.view(batch_size, sent_len, -1)


class Model(nn.Module):
    """
    morpheme prediction and spacing model (a multi-task model)
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            cfg:  config
        """
        super().__init__()
        self.cfg = cfg
        self.conv_layer = ConvLayer(cfg)
        hidden_dim = (cfg.embed_dim * len(self.conv_layer.convs) + cfg.vocab_out) // 2
        setattr(cfg, 'hidden_dim', hidden_dim)
        self.hidden_layer_pos = nn.Sequential(OrderedDict([
            ('feature_dropout', nn.Dropout(p=cfg.hdn_dropout)),
            ('conv2hidden', nn.Linear(cfg.embed_dim * len(self.conv_layer.convs), hidden_dim)),
            ('relu1', nn.ReLU()),
            ('hidden_dropout', nn.Dropout(p=cfg.hdn_dropout)),
            ('hidden2logit', nn.Linear(hidden_dim, cfg.vocab_out)),
        ]))
        self.hidden_layer_spc = nn.Sequential(
            nn.Dropout(p=cfg.hdn_dropout),
            nn.Linear(cfg.embed_dim * len(self.conv_layer.convs), hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.hdn_dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: Tensor):    # pylint: disable=arguments-differ
        """
        Args:
            batch:  batch input
        Returns:
            logits of part-of-speech tags
            logits of right spaces
        """
        use_spc_mask = self.cfg.spc_dropout < 1.0
        features_pos = self.conv_layer(batch, use_spc_mask)
        features_spc = self.conv_layer(batch, False)
        logits_pos = self.hidden_layer_pos(features_pos)
        logits_spc = self.hidden_layer_spc(features_spc)
        return logits_pos, logits_spc

    def save(self, path: str):
        """
        save the model
        Args:
            path:  file path
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        load the model
        Args:
            path:  file path
            conv_layer:  convolution layer
        """
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        if torch.cuda.is_available() and self.cfg.gpu_num >= 0:
            self.cuda(device=self.cfg.gpu_num)
