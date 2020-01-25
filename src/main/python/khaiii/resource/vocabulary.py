# -*- coding: utf-8 -*-


"""
vocabulary library for torchtext
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from collections import Counter
import logging
import os

from torchtext.vocab import Vocab

from khaiii.resource.morphs import TAGS


#############
# variables #
#############
_LOG = logging.getLogger(__name__)


#########
# types #
#########
class StrToIdx(dict):
    """
    string to index dictionary which can handle unknown entry
    """
    def __init__(self, unk: str = None):
        """
        Args:
            unk:  unknown entry
        """
        super().__init__()
        self.unk = unk

    def __getitem__(self, key: str) -> int:
        """
        Args:
            key:  string key
        Returns:
            integer value
        """
        try:
            return super().__getitem__(key)
        except KeyError as key_err:
            if self.unk:
                return super().__getitem__(self.unk)
            raise key_err


class Vocabulary(Vocab):
    """
    input/output vocabulary base class
    """
    def __init__(self, path: str, min_freq: int = 1, unk: str = None):    # pylint: disable=super-init-not-called
        """
        not calling suer().__init__() method is intentional
        Args:
            path:  file path
            min_freq:  minimum frequency
            unk:  unknown word
        """
        self.stoi = StrToIdx(unk)
        self.itos = []
        self._load(path, min_freq)
        assert not unk or unk in self.stoi, 'unknown word {} is not in dictionary'.format(unk)
        self.UNK = unk    # pylint: disable=invalid-name
        assert len(self.stoi) == len(self.itos), \
               'stoi({}) and itos({}) lengths are different'.format(len(self.stoi), len(self.itos))

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, key):
        """
        Args:
            key:  key
        Returns:
            index for string key
            word for integer key
        """
        if isinstance(key, str):
            return self.stoi[key]
        if isinstance(key, int):
            return self.itos[key]
        raise ValueError('invalid key type: {}'.format(type(key)))

    def _load(self, path: str, min_freq: int):
        """
        load vocabulary from file
        Args:
            path:  file path
            min_freq:  minimum frequency
        """
        cutoff_num = 0
        for line in open(path, 'r', encoding='UTF-8'):
            line = line.rstrip('\r\n')
            if not line:
                continue
            try:
                entry, freq = line.split('\t')
            except ValueError:
                entry, freq = line, '0'
            freq = int(freq)
            if 0 < freq < min_freq:    # zero frequencies are not cut-offed
                cutoff_num += 1
                continue
            self.stoi[entry] = len(self.stoi)
            self.itos.append(entry)
        _LOG.info('%s: %d entries, %d cutoff', os.path.basename(path), len(self), cutoff_num)

    def debug_itos(self) -> str:
        """
        make debug string for self.itos
        Returns:
            debug string of self.itos
        """
        idx_vals = list(enumerate(self.itos[:10]))
        middle = len(self.itos) // 2
        idx_vals.extend(enumerate(self.itos[middle:middle+10], start=middle))
        last = len(self.itos) - 10
        idx_vals.extend(enumerate(self.itos[last:], start=last))
        return ', '.join(['{}: {}'.format(i, v) for i, v in idx_vals])


class VocabIn(Vocabulary):
    """
    input vocabulary
    """
    # special characters
    specials = ['<pad>', '<unk>', '<w>', '</w>']

    def __init__(self, path: str, min_freq: int = 2):
        """
        Args:
            path:  file path
            min_freq:  minimum frequency
        """
        super().__init__(path, min_freq, '<unk>')
        assert self.itos[:len(self.specials)] == self.specials, \
               'specials are not the first index: {}'.format(self.itos[:len(self.specials)])
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug('==== itos: %s', self.debug_itos())
            _LOG.debug('==== UNK: %d: %s', self[self.UNK], self.UNK)
        _LOG.info('input vocabulary: %d', len(self))

    @classmethod
    def save(cls, cnt: Counter, path: str, min_freq: int = 1):
        """
        save input vocabulary to file
        Args:
            cnt:  frequencies (Counter object)
            path:  file path
            min_freq:  minimum frequency
        """
        with open(path, 'w', encoding='UTF-8') as fout:
            for special in cls.specials:
                print('{}\t0'.format(special), file=fout)
            for char, freq in sorted(cnt.items()):
                if freq < min_freq:
                    continue
                print('{}\t{}'.format(char, freq), file=fout)


class VocabOut(Vocabulary):
    """
    output vocabulary
    """
    # special tags (simple tags)
    specials = ['B-{}'.format(tag) for tag in TAGS] + ['I-{}'.format(tag) for tag in TAGS]

    def __init__(self, path: str):
        """
        Args:
            path:  file path
            min_freq:  minimum frequency
        """
        super().__init__(path)
        self.stoi['<pad>'] = len(self.stoi)
        self.itos.append('<pad>')
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug('==== itos: %s', self.debug_itos())
        _LOG.info('output vocabulary: %d', len(self))

    @classmethod
    def save(cls, cnt: Counter, path: str):
        """
        save output vocabulary to file
        Args:
            cnt:  frequencies (Counter object)
            path:  file path
        """
        with open(path, 'w', encoding='UTF-8') as fout:
            for special in cls.specials:
                print(special, file=fout)
                del cnt[special]
            for tag in sorted(cnt.keys()):
                print(tag, file=fout)
