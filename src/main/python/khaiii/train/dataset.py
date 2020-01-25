# -*- coding: utf-8 -*-


"""
training dataset for torchtext
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import copy
import logging
import math
import os
from queue import Queue, Empty
from threading import Thread
from typing import Dict, List, TextIO, Tuple

import torch
from torch import Tensor    # pylint: disable=no-name-in-module
from torchtext.data import BucketIterator, Dataset, Example, Field, Iterator
from torchtext.vocab import Vocab
from tqdm import tqdm

from khaiii.resource.resource import Resource
from khaiii.resource.vocabulary import VocabIn, VocabOut


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


#########
# types #
#########
class Morph:
    """
    morpheme
    """
    def __init__(self, lex: str, tag: str = 'O', beg: int = -1, end: int = -1):
        """
        Args:
            lex:  lexical
            tag:  part-of-speech tag
            beg:  begin position
            end:  end position
        """
        self.lex = lex
        self.tag = tag
        self.beg = beg
        self.end = end

    def __str__(self):
        return '{}/{}'.format(self.lex, self.tag)

    def __len__(self):
        return self.end - self.beg


class Word:
    """
    word
    """
    def __init__(self, raw: str):
        """
        Args:
            raw:  raw word
        """
        self.raw = raw
        self.tags = []    # output tags for each character
        self.res_chrs = raw    # concatenation of characters of restored morphemes
        self.res_tags = []    # list of output tags from restored morphemes for each character
        self.morphs = [Morph(c, 'O', i, i+1) for i, c in enumerate(raw)]

    def __str__(self):
        return '{}\t{}'.format(self.raw, ' '.join([str(x) for x in self.morphs]))

    def __eq__(self, other: 'Word'):
        """
        assume equal if two morphologically analyzed results are same.
        use this equal operator when evaluate valid/test dataset
        Args:
            other:  other object
        """
        return self.res_chrs == other.res_chrs and self.res_tags == other.res_tags

    def set_tags(self, tags: List[str], restore_dic: Dict[str, str] = None):
        """
        apply predicted output labels to this word with restore dictionary
        Args:
            tags:  output labels for each character
            restore_dic:  restore dictionary
        """
        if not restore_dic:
            tags = [x.split(':', 1)[0] for x in tags]
        self.tags = tags
        assert len(self.raw) == len(self.tags)    # 음절수와 태그수는 동일해야 한다.
        self.morphs = self._make_morphs(restore_dic)

    def _make_morphs(self, restore_dic: Dict[str, str] = None):
        """
        make morphemes from characters with B-/I- tags
        Args:
            restore_dic:  restore dictionary
        """
        if not self.tags:
            return []

        self._restore(restore_dic)

        morphs = []
        for beg, (lex, iob_tag) in enumerate(zip(self.res_chrs, self.res_tags)):
            try:
                iob, tag = iob_tag.rsplit('-', 1)
            except ValueError as val_err:
                _LOG.error('invalid char/tag: %s/%s in [%s] %s', lex, iob_tag, self.res_chrs,
                           self.res_tags)
                raise val_err
            if iob == 'B' or not morphs or morphs[-1].tag != tag:
                morphs.append(Morph(lex, tag, beg, beg+1))
            elif iob == 'I':
                if morphs[-1].tag == tag:
                    morphs[-1].lex += lex
                    morphs[-1].end += len(lex)
                else:
                    _LOG.debug('tag is different between B and I: %s vs %s', morphs[-1].tag, tag)
                    morphs.append(Morph(lex, tag, beg, beg+1))
            else:
                raise ValueError('invalid IOB tag: {}/{} in [{}] {}'.format \
                                 (lex, iob_tag, self.res_chrs, self.res_tags))
        return morphs

    def _restore(self, restore_dic: Dict[str, str]):
        """
        restore morphemes with dictionary
        Args:
            restore_dic:  restore dictionary
        """
        if not restore_dic:
            self.res_chrs = self.raw
            self.res_tags = self.tags
            return

        res_chrs = []
        self.res_tags = []
        for char, tag in zip(self.raw, self.tags):
            if ':' in tag:
                key = '{}/{}'.format(char, tag)
                if key in restore_dic:
                    for char_tag in restore_dic[key].split():
                        res_chr, res_tag = char_tag.rsplit('/', 1)
                        res_chrs.append(res_chr)
                        self.res_tags.append(res_tag)
                    continue
                _LOG.debug('mapping not found: %s/%s', char, tag)
                tag, _ = tag.split(':', 1)
            res_chrs.append(char)
            self.res_tags.append(tag)
        self.res_chrs = ''.join(res_chrs)


class Sent(Example):
    """
    single training example
    """
    def __init__(self, words: List[Tuple[str, str]]):
        """
        Args:
            words:  list of words. word is pair of raw and tags
        """
        self.words = []
        self.char = []
        self.tag = []
        self.left_spc = []
        self.right_spc = []
        for raw, tag in words:
            self.words.append(Word(raw))
            self.char.extend(list(raw))
            self.tag.extend(tag.split())
            self.left_spc.extend([1, ] + [0, ] * (len(raw)-1))
            self.right_spc.extend([0, ] * (len(raw)-1) + [1, ])
        assert len(self.char) == len(self.tag), \
               f'char and tag len diff: {len(self.char)} vs {len(self.tag)}'
        assert len(self.tag) == len(self.left_spc), \
               f'tag and left spc len diff: {len(self.tag)} {len(self.left_spc)}'
        assert len(self.left_spc) == len(self.right_spc), \
               f'left and right spc len diff: {len(self.tag)} {len(self.left_spc)}'

    def __str__(self):
        return '\n'.join([f'char: {self.char}', f'tag: {self.tag}', f'left_spc: {self.left_spc}',
                          f'right_spc: {self.right_spc}'])

    def copy(self):
        """
        shallow copy for evaluation
        Returns:
            copied object
        """
        other = copy.copy(self)
        other.words = [copy.copy(w) for w in self.words]
        return other

    def set_tags(self, tags: List[str] = None, restore_dic: Dict[str, str] = None):
        """
        set whole tags(output labels) in sentence and restore morphemes
        Args:
            tags:  output labels
            restore_dic:  restore dictionary
        """
        if not tags:
            tags = self.tag
        total_char_num = 0
        for word in self.words:
            word.set_tags(tags[total_char_num:total_char_num + len(word.raw)], restore_dic)
            total_char_num += len(word.raw)
        assert total_char_num == len(tags)


class CharField(Field):
    """
    characters field
    """
    def __init__(self, vocab: VocabIn, window: int):
        """
        Args:
            window:  window size of context
            vocab:  input vocabulary
        """
        super().__init__(batch_first=True, postprocessing=self.postprocess)
        self.window = window
        self.vocab = vocab

    def itos(self, tensor: Tensor) -> List[str]:
        """
        transform list of numbers(index) into list of strings(chars)
        Args:
            tensor:  list of numbers(index)
        Returns:
            list of strings(chars)
        """
        # since 1st rank items are context(list), print only middle characters in list
        return ', '.join([self.vocab.itos[c[self.window]] for c in tensor])

    def postprocess(self, batch: List[List[int]], vocab: Vocab) -> List[List[int]]:    # pylint: disable=unused-argument
        """
        Args:
            batch:  batch (list of field values)
            vocab:  vocabuary
        """
        new_batch = []
        for chars in batch:
            new_chars = []
            for idx, char in enumerate(chars):
                if char == 0:
                    new_chars.append([0, ] * (2 * self.window + 1))
                    continue
                if idx - self.window < 0:
                    left = 0
                    context = [0, ] * (self.window - idx)
                else:
                    left = idx - self.window
                    context = []
                context.extend(chars[left:idx+self.window+1])
                if idx + self.window+1 > len(chars):
                    context.extend([0, ] * (idx + self.window+1 - len(chars)))
                new_chars.append(context)
            new_batch.append(new_chars)
        return new_batch

    def vocab_size(self):
        """
        Returns:
            vocabulary size
        """
        return len(self.vocab.itos)


class TagField(Field):
    """
    tags field
    """
    def __init__(self, vocab: VocabOut, window: int):
        """
        Args:
            window:  window size of context
            vocab:  output vocabulary
        """
        super().__init__(batch_first=True, is_target=True)
        self.window = window
        self.vocab = vocab

    def itos(self, tensor: Tensor) -> List[str]:
        """
        transform list of numbers(index) into list of strings(chars)
        Args:
            tensor:  list of numbers(index)
        Returns:
            list of strings(chars)
        """
        return ', '.join([self.vocab[c] for c in tensor])

    def vocab_size(self):
        """
        Returns:
            vocabulary size
        """
        return len(self.vocab)-1    # without padding (the last index)


class LeftSpcField(Field):
    """
    left spaces field
    """
    def __init__(self, window: int):
        """
        Args:
            window:  window size of context
        """
        super().__init__(batch_first=True, use_vocab=False, pad_token=-1,
                         postprocessing=self.postprocess)
        self.window = window

    def postprocess(self, batch: List[List[int]], vocab: Vocab) -> List[List[int]]:    # pylint: disable=unused-argument
        """
        Args:
            batch:  batch (list of field values)
            vocab:  vocabuary
        """
        new_batch = []
        for left_spcs in batch:
            new_left_spcs = []
            for idx, left_spc in enumerate(left_spcs):
                if left_spc == -1:
                    new_left_spcs.append([0, ] * (2 * self.window + 1))
                    continue
                left_spc_context = []
                for jdx in range(idx, max(idx-self.window-1, -1), -1):
                    if left_spcs[jdx] == 1:
                        left_spc_context.insert(0, 1)
                        break
                    left_spc_context.insert(0, 0)
                if len(left_spc_context) < self.window+1:
                    left_spc_context = ([0, ] * (self.window+1 - len(left_spc_context))
                                        + left_spc_context)
                left_spc_context.extend([0, ] * self.window)
                new_left_spcs.append(left_spc_context)
            new_batch.append(new_left_spcs)
        return new_batch


class RightSpcField(Field):
    """
    right spaces field
    """
    def __init__(self, window: int):
        """
        Args:
            window:  window size of context
        """
        super().__init__(batch_first=True, use_vocab=False, pad_token=-1,
                         postprocessing=self.postprocess)
        self.window = window

    def postprocess(self, batch: List[List[int]], vocab: Vocab) -> List[List[int]]:    # pylint: disable=unused-argument
        """
        Args:
            batch:  batch (list of field values)
            vocab:  vocabuary
        """
        new_batch = []
        for right_spcs in batch:
            new_right_spcs = []
            for idx, right_spc in enumerate(right_spcs):
                if right_spc == -1:
                    new_right_spcs.append([0, ] * (2 * self.window + 1))
                    continue
                right_spc_context = []
                for jdx in range(idx, min(idx+self.window+1, len(right_spcs)+1)):
                    if right_spcs[jdx] == 1:
                        right_spc_context.append(1)
                        break
                    right_spc_context.append(0)
                if len(right_spc_context) < self.window+1:
                    right_spc_context.extend([0, ] * (self.window+1 - len(right_spc_context)))
                right_spc_context = ([0, ] * self.window) + right_spc_context
                new_right_spcs.append(right_spc_context)
            new_batch.append(new_right_spcs)
        return new_batch

    @classmethod
    def for_criterion(cls, spc_batch: Tensor, tag_batch: Tensor, window: int,
                      device: torch.device = None) -> Tensor:    # pylint: disable=no-member
        """
        make tensor for criterion from space context to space output
        Args:
            spc_batch:  space context
            tag_batch:  tag output
            window:  window size
            device:  device for tensor
        Returns:
            space output
        """
        space_output = []
        for batch_idx, sent in enumerate(spc_batch):
            for char_idx, spc in enumerate(sent):
                if tag_batch[batch_idx][char_idx] == FIELDS['tag'].vocab.stoi['<pad>']:
                    space_output.append(-100)
                else:
                    space_output.append(spc[window].item())
        return torch.tensor(space_output, device=device)    # pylint: disable=not-callable


class BatchIter:
    """
    multi-core batch iterator
    """
    def __init__(self, data: Dataset, batch_size: int, device: torch.device = None):    # pylint: disable=no-member
        """
        Args:
            data:  dataset
            batch_size:  batch_size
            device:  device for tensor
        """
        self.batches = BucketIterator(data, batch_size, device=device, shuffle=True)
        self.queue = Queue(maxsize=10)
        self.worker = Thread(target=self.run)
        self.worker.start()

    def __len__(self):
        return math.ceil(len(self.batches.dataset) / self.batches.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(10):
            try:
                return self.queue.get(timeout=1)
            except Empty:
                if not self.worker.is_alive():
                    break
        raise StopIteration()

    def run(self):
        """
        run iterator thread
        """
        for batch in self.batches:
            self.queue.put(batch)


class SentIter:
    """
    sentence iterator
    """
    def __init__(self, data: Dataset, device: torch.device = None):    # pylint: disable=no-member
        """
        Args:
            data:  dataset
            device:  device for tensor
        """
        self.sents = Iterator(data, 1, device=device, train=False, sort=False)
        self.queue = Queue(maxsize=10)
        self.worker = Thread(target=self.run)
        self.worker.start()

    def __len__(self):
        return len(self.sents.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(10):
            try:
                return self.queue.get(timeout=1)
            except Empty:
                if not self.worker.is_alive():
                    break
        raise StopIteration()

    def run(self):
        """
        run iterator thread
        """
        for tensor, sent in zip(self.sents, self.sents.data()):
            self.queue.put((tensor, sent))


#############
# variables #
#############
FIELDS = {}


#############
# functions #
#############
def load(fin: TextIO, window: int, rsc: Resource) -> Dataset:
    """
    load dataset from file
    Args:
        fin:  input file
        min_freq:  minimum freq. for characters vocab.
        window:  window size of context
        rsc:  Resource
    Returns:
        Dataset object
    """
    lines = fin.readlines()
    sents = []
    words = []
    for line in tqdm(lines, os.path.basename(fin.name), len(lines), mininterval=1, ncols=100):
        line = line.rstrip('\r\n')
        if not line:
            if words:
                sents.append(Sent(words))
                words = []
            continue
        raw, tags = line.split('\t')
        words.append((raw, tags))

    FIELDS['char'] = CharField(rsc.vocab_in, window)
    FIELDS['tag'] = TagField(rsc.vocab_out, window)
    FIELDS['left_spc'] = LeftSpcField(window)
    FIELDS['right_spc'] = RightSpcField(window)
    data = Dataset(sents, FIELDS.items())

    return data
