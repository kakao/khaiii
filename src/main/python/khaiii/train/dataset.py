# -*- coding: utf-8 -*-


"""
corpus for training
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import itertools
import logging
import os
import random
from typing import Dict, List, TextIO, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from khaiii.resource.resource import Resource
from khaiii.train.sentence import PosSentence, PosWord


#########
# types #
#########
class PosSentTensor(PosSentence):
    """
    tensor transformable sentence
    """
    def __init__(self, raw_sent: str = ''):
        super().__init__(raw_sent)
        if raw_sent:
            self.init_pos_tags()

    def __len__(self):
        # RNN에서 길이가 긴 것으로 정렬하기위한 문장의 길이 = 음절 갯수 + 문장 경계 + 어절 경계
        if self.words:
            return sum([len(w) for w in self.words]) + len(self.words) + 1
        if self.pos_tagged_words:
            return sum([len(w.raw) for w in self.pos_tagged_words]) + len(self.pos_tagged_words) + 1
        return 0

    @classmethod
    def to_tensor(cls, arr: List, gpu_num: int = -1) -> Tensor:
        """
        Args:
            arr:  array to convert
            gpu_num:  GPU device number. default: -1 for CPU
        Returns:
            tensor
        """
        # pylint: disable=no-member
        device = torch.device('cuda', gpu_num) if torch.cuda.is_available() and gpu_num >= 0 \
                                               else torch.device('cpu')
        return torch.tensor(arr, device=device)    # pylint: disable=not-callable

    def make_contexts(self, window: int) -> List[List[str]]:
        """
        각 음절 별로 좌/우 window 크기 만큼 context를 만든다.
        Args:
            window:  left/right window size
        Returns:
            contexts
        """
        chars = [c for w in self.words for c in w]
        chars_len = len(chars)
        chars_padded = ['', ] * window + chars + ['', ] * window
        contexts = [chars_padded[idx-window:idx+window+1]
                    for idx in range(window, chars_len + window)]
        return contexts

    @classmethod
    def _flatten(cls, list_of_lists):
        """
        flatten one level of nesting
        Args:
            list_of_lists:  list of lists
        Returns:
            flattened list
        """
        return list(itertools.chain.from_iterable(list_of_lists))

    def make_left_spc_masks(self, window: int, left_vocab_id: int, spc_dropout: float) \
            -> List[List[int]]:
        """
        각 음절 별로 좌/우 window 크기 만큼 context를 만든다.
        Args:
            window:  left/right window size
            left_vocab_id:  vocabulary ID for '<w>'
            spc_dropout:  space dropout rate
        Returns:
            left space masks
        """
        def _filter_left_spc_mask(left_spc_mask):
            """
            중심 음절로부터 첫번째 왼쪽 공백만 남기고 나머지는 제거한다.
            Args:
                left_spc_mask:  왼쪽 공백 마스크
            """
            for idx in range(window, -1, -1):
                if left_spc_mask[idx] == left_vocab_id:
                    if random.random() < spc_dropout:
                        left_spc_mask[idx] = 0
                    for jdx in range(idx-1, -1, -1):
                        left_spc_mask[jdx] = 0
                    break

        left_spcs = self._flatten([[left_vocab_id, ] + [0, ] * (len(word)-1)
                                   for word in self.words])
        left_padded = [0, ] * window + left_spcs + [0, ] * window
        left_spc_masks = [left_padded[idx-window:idx+1] + [0, ] * window
                          for idx in range(window, len(left_spcs) + window)]
        for left_spc_mask in left_spc_masks:
            _filter_left_spc_mask(left_spc_mask)
        return left_spc_masks

    def make_right_spc_masks(self, window: int, right_vocab_id: int, spc_dropout: float) \
            -> List[List[int]]:
        """
        각 음절 별로 좌/우 window 크기 만큼 context를 만든다.
        Args:
            window:  left/right window size
            right_vocab_id:  vocabulary ID for '</w>'
            spc_dropout:  space dropout rate
        Returns:
            right space masks
        """
        def _filter_right_spc_mask(right_spc_mask):
            """
            중심 음절로부터 첫번째 오른쪽 공백만 남기고 나머지는 제거한다.
            Args:
                right_spc_mask:  오른쪽 공백 마스크
            """
            for idx in range(window, len(right_spc_mask)):
                if right_spc_mask[idx] == right_vocab_id:
                    if random.random() < spc_dropout:
                        right_spc_mask[idx] = 0
                    for jdx in range(idx+1, len(right_spc_mask)):
                        right_spc_mask[jdx] = 0
                    break

        right_spcs = self._flatten([[0, ] * (len(word)-1) + [right_vocab_id, ]
                                    for word in self.words])
        right_padded = [0, ] * window + right_spcs + [0, ] * window
        right_spc_masks = [[0, ] * window + right_padded[idx:idx+window+1]
                           for idx in range(window, len(right_spcs) + window)]
        for right_spc_mask in right_spc_masks:
            _filter_right_spc_mask(right_spc_mask)
        return right_spc_masks

    def get_contexts(self, cfg: Namespace, rsc: Resource) -> List[List[int]]:
        """
        문맥을 반환하는 메서드
        Args:
            cfg:  config
            rsc:  Resource object
        Returns
            문맥 리스트. shape: [(문장 내 음절 길이), (문맥의 크기)]
        """
        contexts = self.make_contexts(cfg.window)
        return [[rsc.vocab_in[c] for c in context] for context in contexts]

    def get_spc_masks(self, cfg: Namespace, rsc: Resource, do_spc_dropout: bool) \
                -> Tuple[List[List[int]], List[List[int]]]:
        """
        공백 마스킹 벡터를 반환하는 메소드
        Args:
            cfg:  config
            rsc:  Resource object
            do_spc_dropout:  공백 마스크 시 dropout 적용 여부
        Returns
            좌측 공백 마스킹 벡터. shape: [(문장 내 음절 길이), (문맥의 크기)]
            우측 공백 마스킹 벡터. shape: [(문장 내 음절 길이), (문맥의 크기)]
        """
        spc_dropout = cfg.spc_dropout if do_spc_dropout else 0.0
        left_spc_masks = self.make_left_spc_masks(cfg.window, rsc.vocab_in['<w>'], spc_dropout)
        right_spc_masks = self.make_right_spc_masks(cfg.window, rsc.vocab_in['</w>'], spc_dropout)
        return left_spc_masks, right_spc_masks

    def get_labels(self, rsc: Resource) -> List[int]:
        """
        레이블(출력 태그)를 반환하는 메서드
        Args:
            rsc:  Resource object
        Returns
            레이블 리스트. shape: [(문장 내 음절 길이), ]
        """
        return [rsc.vocab_out[tag] for pos_word in self.pos_tagged_words for tag in pos_word.tags]

    def get_spaces(self) -> List[int]:
        """
        음절 별 공백 여부를 반환하는 메서드
        Returns
            공백 여부 리스트. shape: [(문장 내 음절 길이), ]
        """
        spaces = []
        for word in self.words:
            spaces.extend([0, ] * (len(word)-1))
            spaces.append(1)
        return spaces


class PosDataset:
    """
    part-of-speech tag dataset
    """
    def __init__(self, cfg: Namespace, restore_dic: Dict[str, str], fin: TextIO):
        """
        Args:
            cfg:  config
            restore_dic:  restore dictionary
            fin:  input file
        """
        self.cfg = cfg
        self.fin = fin
        self.sents = []
        self.sent_idx = -1
        self._load(restore_dic)

    def __str__(self):
        return '<PosDataset: file: {}, sents: {}, sent_idx: {}>'.format(
            os.path.basename(self.fin.name), len(self.sents), self.sent_idx)

    def _load(self, restore_dic: dict):
        """
        load data file
        Args:
            restore_dic:  restore dictionary
        """
        sent = PosSentTensor()
        lines = self.fin.readlines()
        for line in tqdm(lines, os.path.basename(self.fin.name), len(lines), mininterval=1,
                         ncols=100):
            line = line.rstrip('\r\n')
            if not line:
                if sent and sent.pos_tagged_words:
                    sent.set_raw_by_words()
                    self.sents.append(sent)
                sent = PosSentTensor()
                continue
            raw, tags = line.split('\t')
            pos_word = PosWord(raw)
            pos_word.set_pos_result(tags.split(), restore_dic)
            sent.pos_tagged_words.append(pos_word)
        logging.info('%s: %d sentences', os.path.basename(self.fin.name), len(self.sents))

    def __iter__(self):
        self.sent_idx = -1
        random.shuffle(self.sents)
        return self

    def __next__(self):
        self.sent_idx += 1
        if self.sent_idx >= len(self.sents):
            raise StopIteration()
        return self.sents[self.sent_idx]

    def __len__(self):
        return len(self.sents)
