#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Sejong corpus parser
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser
import logging
import os
import re
import sys
from typing import Iterator, TextIO
from unicodedata import normalize as norm


#############
# constants #
#############
TAG_SET = {    # sejong tag set
    'NNG', 'NNP', 'NNB', 'NP', 'NR',    # 체언
    'VV', 'VA', 'VX', 'VCP', 'VCN',    # 용언
    'MM', 'MAG', 'MAJ',    # 수식언
    'IC',    # 독립언
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',    # 관계언
    'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR',    # 의존형태
    'SF', 'SP', 'SS', 'SE', 'SO', 'SL', 'SH', 'SW', 'SWK', 'SN',    # 기호
    'ZN', 'ZV', 'ZZ',    # 분석 불능
}

WORD_ID_PTN = re.compile(r'^[0-9A-Z_]{4}\d{4}-\d{7,8}')

SENT_OPEN_TAGS = ['<head>', '<p>', '<l>']    # sentence open tags at written corpus

# sentence close tags at written corpus
SENT_CLOSE_TAGS = [tag[0] + '/' + tag[1:] for tag in SENT_OPEN_TAGS]

_OPEN_TAGS_IN_WRITTEN_SENT = {'<date>', }    # open tags in sentence at written corpus

# close tags in sentence at written corpus
_CLOSE_TAGS_IN_WRITTEN_SENT = {tag[0] + '/' + tag[1:] for tag in _OPEN_TAGS_IN_WRITTEN_SENT}

# open and close tags in sentence at written corpus
_TAGS_IN_WRITTEN_SENT = _OPEN_TAGS_IN_WRITTEN_SENT | _CLOSE_TAGS_IN_WRITTEN_SENT


#########
# types #
#########
class ParseError(Exception):
    """
    error occurred while parsing corpus
    """


class Morph:    # pylint: disable=too-few-public-methods
    """
    morpheme
    """
    def __init__(self, lex: str = '', tag: str = ''):
        if ' ' in lex:
            raise ParseError('space in raw morph: %s' % lex)
        self.lex = lex    # lexical form
        self.tag = tag    # part-of-speech tag

    def __str__(self):
        return '{}/{}'.format(self.lex, self.tag)

    def __eq__(self, other: 'Morph'):
        return self.lex == other.lex and self.tag == other.tag

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def parse(cls, token_str: str, file_name: str = '', line_num: int = 0):
        """
        parse token string
        Args:
            token_str:  morpheme/tag string
            file_name:  file name
            line_num:  line number
        Returns:
            Morph object
        """
        def _raise(err_msg: str, file_name: str = None, line_num: int = 0):
            """
            에러 메세지를 이용해 파싱 에러를 발생시킨다.
            Arguments:
                err_msg:  에러 메세지
                file_name:  파일명
                line_num:  라인 번호
            """
            file_pfx = '{}({}) '.format(file_name, line_num) if file_name and line_num else ''
            raise ParseError('{}{}'.format(file_pfx, err_msg))

        morph = Morph()
        morph.lex, morph.tag = token_str.rsplit('/', 1)
        if not morph.lex:
            _raise('no text in morpheme: {}'.format(token_str), file_name, line_num)
        if ' ' in morph.lex:
            _raise('space in raw morph: {}'.format(token_str), file_name, line_num)
        if morph.tag not in TAG_SET:
            _raise('invalid tag: {} in {}'.format(morph.tag, token_str), file_name, line_num)
        for char in morph.lex:
            if 0x1100 <= ord(char) < 0x1200:
                _raise('Hangul Jamo character: "{}" in {}'.format(char, token_str), file_name,
                       line_num)
        return morph


class Word:
    """
    word(어절)
    """
    def __init__(self):
        self.wid = ''    # word ID
        self.raw = ''    # raw word string
        self.morphs = []    # morpheme list

    def __str__(self):
        return '{}\t{}\t{}'.format(self.wid, self.raw, ' + '.join([str(m) for m in self.morphs]))

    def __eq__(self, other: 'Word'):
        if self.raw != other.raw:
            return False
        if len(self.morphs) != len(other.morphs):
            return False
        for my_morph, other_morph in zip(self.morphs, other.morphs):
            if my_morph != other_morph:
                return False
        return True

    def morph_str(self) -> str:
        """
        make morpheme string
        Returns:
            morpheme string
        """
        return ' + '.join([str(_) for _ in self.morphs])

    @classmethod
    def parse(cls, line: str, file_name: str, line_num: int) -> 'Word':
        """
        parse word(EoJeol) with single line
        Args:
            line:  line
            file_name:  file name
            line_num:  line number
        Returns:
            Word object
        """
        cols = line.split('\t')
        if len(cols) != 3:
            if Sentence.is_tag_in_sent(line):
                return None
            raise ParseError('%s(%d) Invalid line: %s' % (file_name, line_num, line))
        word = Word()
        word.wid, word.raw = cols[0], cols[1]
        if ' ' in word.raw:
            raise ParseError('%s(%d) space in raw word: %s' % (file_name, line_num, line))
        try:
            word.morphs = [Morph.parse(word_tag, file_name, line_num) for word_tag
                           in cols[2].split(' + ')]
        except ValueError as val_err:
            raise ParseError('%s(%d) %s: %s' % (file_name, line_num, val_err, line))
        morphs_raw = ''.join([m.lex for m in word.morphs])
        if (len(word.raw) == len(morphs_raw) and word.raw != morphs_raw and
                norm('NFKD', word.raw) == norm('NFKD', morphs_raw)):
            raise ParseError('%s(%d) raw-morph mismatch: %s' % (file_name, line_num, line))
        return word


class Sentence:
    """
    sentence
    """
    def __init__(self):
        self.words = []    # word list
        self._wids = set()    # word ID set

    def __str__(self):
        words_str = '\n'.join([str(_) for _ in self.words])
        return '# %s\n%s' % (self.raw_str(), words_str)

    def raw_str(self) -> str:
        """
        raw sentence (words)
        Returns:
            raw sentence
        """
        return ' '.join([_.raw for _ in self.words])

    def morph_str(self) -> str:
        """
        make morpheme string
        Returns:
            morpheme string
        """
        return ' + '.join([_.morph_str() for _ in self.words])

    @classmethod
    def is_opening(cls, line: str) -> bool:
        """
        whether sentence opening or not
        Args:
            line:  line
        Returns:
            whether opening or not
        """
        return line in SENT_OPEN_TAGS

    @classmethod
    def is_closing(cls, line: str) -> bool:
        """
        whether sentence closing or not
        Args:
            line:  line
        Returns:
            whether closing or not
        """
        return line in SENT_CLOSE_TAGS

    @classmethod
    def is_tag_in_sent(cls, line: str) -> bool:
        """
        whether tag in sentence or not
        Args:
            line:  line
        Returns:
            whether tag or not
        """
        return line in _TAGS_IN_WRITTEN_SENT

    def is_good_tags(self) -> bool:
        """
        whether all tags in sentence are good(correct) or not
        Returns:
            whether all tags are good
        """
        return not [morph.tag for word in self.words for morph in word.morphs
                    if morph.tag not in TAG_SET]

    def append(self, word: Word):
        """
        append word
        Args:
            word:  Word object
        """
        if word.wid and word.wid in self._wids:
            raise ParseError('duplicated word ID: %s' % word)
        self.words.append(word)
        self._wids.add(word.wid)


#############
# functions #
#############
def sents(fin: TextIO) -> Iterator[Sentence]:
    """
    load from file and return sentences (generator)
    Args:
        fin:  input file object
    Yields:
        Sentence object
    """
    file_name = os.path.basename(fin.name)
    par_errs = []
    sent = None
    has_invalid_word = False
    for line_num, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        if line == '</tei.2>' and 'BTJO0443.txt' in file_name:
            break
        if Sentence.is_opening(line):
            sent = Sentence()
            continue
        elif Sentence.is_closing(line):
            if sent and sent.words and not has_invalid_word:
                if sent.is_good_tags():
                    yield sent
            sent = None
            has_invalid_word = False
            continue
        elif not sent:
            continue
        try:
            word = Word.parse(line, file_name, line_num)
            if word:
                sent.append(word)
        except ParseError as par_err:
            par_errs.append(par_err)
            has_invalid_word = True
    if par_errs:
        for par_err in par_errs:
            logging.error(par_err)
        raise ParseError('there is(are) %d error(s) in file: %s' % (len(par_errs), file_name))


def run():
    """
    run function which is the start point of program
    """
    for sent in sents(sys.stdin):
        print(sent)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='load Sejong corpus and print')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()
