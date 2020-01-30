# -*- coding: utf-8 -*-


"""
part-of-speech tagger
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import json
import logging

import torch
from torch import Tensor
import torch.nn.functional as F

from khaiii.resource.resource import Resource
from khaiii.train import dataset
from khaiii.train.dataset import Sent
from khaiii.train.models import Model


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


#########
# types #
#########
class PosTagger:
    """
    part-of-speech tagger
    """
    def __init__(self, model_dir: str, gpu_num: int = -1):
        """
        Args:
            model_dir:  model dir
            gpu_num:  GPU number to override
        """
        cfg_dict = json.load(open('{}/config.json'.format(model_dir), 'r', encoding='UTF-8'))
        self.cfg = Namespace()
        for key, val in cfg_dict.items():
            setattr(self.cfg, key, val)
        setattr(self.cfg, 'gpu_num', gpu_num)
        self.rsc = Resource(self.cfg)
        self.model = Model(self.cfg)
        self.model.load('{}/model.state'.format(model_dir))
        self.model.eval()

    def tag_batch(self, batch: Tensor, sent: Sent):
        """
        part-of-speech tagging for a batch tensor which is a single sentence
        Args:
            batch:  a tensor of batch
            sent:  Sent object
        """
        tag_vocab = dataset.FIELDS['tag'].vocab.itos
        with torch.no_grad():
            logits, _ = self.model(batch)
            _, predicts = F.softmax(logits[0], dim=1).max(1)
            pred_tags = [tag_vocab[t.item()] for t in predicts]
            _LOG.debug('sent: %s', sent.char)
            _LOG.debug('tags: %s', pred_tags)
            sent.set_tags(pred_tags, self.rsc.restore_dic)

        if _LOG.isEnabledFor(logging.DEBUG):
            raw_nospc = ''.join(sent.char)
            for idx, (tag, pred) in enumerate(zip(sent.tag, predicts)):
                _LOG.debug('[%2d]%s: %5s(%d)', idx, raw_nospc[idx], tag, pred.item())
