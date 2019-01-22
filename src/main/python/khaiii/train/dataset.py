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
import logging
import os
import random
from typing import List, TextIO, Tuple

from torch import LongTensor, Tensor    # pylint: disable=no-member, no-name-in-module
from tqdm import tqdm

from khaiii.resource.resource import PAD_CHR, Resource
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

    def make_labels(self, with_spc: bool) -> List[str]:
        """
        각 음절별로 출력 레이블(태그)를 생성한다.
        Args:
            with_spc:  공백(어절 경계) 포함 여부
        Returns:
            레이블 리스트
        """
        if not with_spc:
            # 문장 경계, 어절 경계 등 가상 음절을 제외하고 순수한 음절들의 레이블
            return [tag for pos_word in self.pos_tagged_words for tag in pos_word.tags]
        labels = [PAD_CHR, ]    # 문장 시작
        for pos_word in self.pos_tagged_words:
            if len(labels) > 1:
                labels.append(PAD_CHR)    # 어절 경계
            labels.extend(pos_word.tags)
        labels.append(PAD_CHR)    # 문장 종료
        return labels

    def make_contexts(self, window: int, spc_dropout: float) -> List[str]:
        """
        각 음절 별로 좌/우 window 크기 만큼 context를 만든다.
        Args:
            window:  left/right window size
            spc_dropout:  space(word delimiter) dropout rate
        Returns:
            contexts
        """
        contexts = []
        for wrd_idx, word in enumerate(self.words):
            for chr_idx, char in enumerate(word):
                left_context = list(reversed(word[:chr_idx]))
                if random.random() >= spc_dropout:
                    left_context.append('<w>')
                for left_word in reversed(self.words[:wrd_idx]):
                    left_context.extend(reversed(left_word))
                    if len(left_context) >= window:
                        break
                if len(left_context) < window:
                    left_context.extend(['<s>', ] * (window - len(left_context)))
                left_context = list(reversed(left_context[:window]))
                assert len(left_context) == window

                right_context = list(word[chr_idx+1:])
                if random.random() >= spc_dropout:
                    right_context.append('</w>')
                for right_word in self.words[wrd_idx+1:]:
                    right_context.extend(list(right_word))
                    if len(right_context) >= window:
                        break
                if len(right_context) < window:
                    right_context.extend(['</s>', ] * (window - len(right_context)))
                right_context = right_context[:window]
                assert len(right_context) == window
                contexts.append(left_context + [char, ] + right_context)
        return contexts

    def to_tensor(self, cfg: Namespace, rsc: Resource, is_train: bool) -> Tuple[Tensor, Tensor]:
        """
        문장 내에 포함된 전체 음절들과 태그를 모델의 forward 메소드에 넣을 수 있는 텐서로 변환한다.
        Args:
            cfg:  config
            rsc:  Resource object
            is_train:  whether is train or not
        Returns:
            labels tensor
            contexts tensor
        """
        # 차원: [문장내 음절 갯수, ]
        label_nums = [rsc.vocab_out[l] for l in self.make_labels(False)]
        labels_tensor = LongTensor(label_nums)
        # 차원: [문장내 음절 갯수 x context 크기]
        spc_dropout = cfg.spc_dropout if is_train else 0.0
        context_nums = [[rsc.vocab_in[c] for c in context] \
                        for context in self.make_contexts(cfg.window, spc_dropout)]
        contexts_tensor = LongTensor(context_nums)
        return labels_tensor, contexts_tensor

    def make_chars(self) -> List[str]:
        """
        문장 내 포함된 음절들을 만든다. 문장 경계 및 어절 경계를 포함한다.
        Returns:
            음절의 리스트
        """
        chars = ['<s>', ]    # 문장 시작
        for word in self.words:
            if len(chars) > 1:
                chars.append('<w>')    # 어절 경계
            chars.extend(word)
        chars.append('</s>')    # 문장 종료
        return chars


class PosDataset:
    """
    part-of-speech tag dataset
    """
    def __init__(self, cfg: Namespace, restore_dic: dict, fin: TextIO):
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
