# -*- coding: utf-8 -*-


"""
parsing module for morphologically analyzed results
TODO(jamie): duplicated to Morph class in sejong_corpus module
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from typing import List


#############
# constants #
#############
# all tag set for index -> tag mapping
TAGS = sorted(['EC', 'EF', 'EP', 'ETM', 'ETN', 'IC', 'JC', 'JKB', 'JKC', 'JKG',
               'JKO', 'JKQ', 'JKS', 'JKV', 'JX', 'MAG', 'MAJ', 'MM', 'NNB', 'NNG',
               'NNP', 'NP', 'NR', 'SE', 'SF', 'SH', 'SL', 'SN', 'SO', 'SP',
               'SS', 'SW', 'SWK', 'VA', 'VCN', 'VCP', 'VV', 'VX', 'XPN', 'XR',
               'XSA', 'XSN', 'XSV', 'ZN', 'ZV', 'ZZ', ])
# "B-" possible tags
B_TAGS = sorted(['EP', 'IC', 'JKB', 'JX', 'MAG', 'MM', 'NNB', 'NNG', 'NNP', 'NP',
                 'NR', 'SE', 'SF', 'SN', 'SO', 'SP', 'SS', 'SW', 'SWK', 'XPN',
                 'XR', 'XSN', 'ZN', ])
TAG_SET = {tag: num for num, tag in enumerate(TAGS, start=1)}    # tag -> index mapping

# virtual morphemes for error patch
WORD_DELIM_STR = '_'    # word delimiter(space) symbol
SENT_DELIM_STR = '|'    # sentence boundary symbol
WORD_DELIM_NUM = -1    # word delimiter(space) number
SENT_DELIM_NUM = -2    # sentence boundary number


#########
# types #
#########
class ParseError(Exception):
    """
    errors occurred when parsing morphologically analyzed results
    """


class Morph:
    """
    morpheme
    """
    def __init__(self, lex: str, tag: str):
        """
        Arguments:
            lex:  lexical form
            tag:  part-of-speech tag
        """
        self.lex = lex
        self.tag = tag

    def __str__(self):
        if not self.tag:
            return self.lex
        return '{}/{}'.format(self.lex, self.tag)

    def is_word_delim(self) -> bool:
        """
        whether is word delimiter or not
        Returns:
            whether is word delimiter
        """
        return not self.tag and self.lex == WORD_DELIM_STR

    def is_sent_delim(self) -> bool:
        """
        whether is sentence delimiter or not
        Returns:
            whether is sentence delimiter
        """
        return not self.tag and self.lex == SENT_DELIM_STR

    @classmethod
    def to_str(cls, morphs: List['Morph']) -> str:
        """
        make string for list of Morph objects
        Arguments:
            morphs:  list of Morph objects
        Returns:
            string
        """
        return ' + '.join([str(m) for m in morphs])

    @classmethod
    def parse(cls, morphs_str: str) -> List['Morph']:
        """
        parse morphologically analyzed results and return the analyzed results
        Arguments:
            morphs_str:  morphlogically analyzed string. ex: "제이미/NNP + 는/JKS"
        Returns:
            list of Morph objects
        """
        if not morphs_str:
            raise ParseError('empty to parse')
        return [cls._parse_one(m) for m in morphs_str.split(' + ')]

    @classmethod
    def _parse_one(cls, morph_str: str) -> 'Morph':
        """
        parse a single morpheme string
        Arguments:
            morph_str:  morpheme string
        Returns:
            Morph object
        """
        if ' ' in morph_str:
            raise ParseError('space in morph')
        try:
            if morph_str in [WORD_DELIM_STR, SENT_DELIM_STR]:
                return Morph(morph_str, '')
            lex, tag = morph_str.rsplit('/', 1)
        except ValueError:
            raise ParseError('invalid morpheme string format')
        if not lex:
            raise ParseError('no lexical in morpheme string')
        if not tag:
            raise ParseError('no pos tag in morpheme string')
        if tag not in TAG_SET:
            raise ParseError('invalid pos tag: {}'.format(tag))
        return Morph(lex, tag)


#############
# functions #
#############
def mix_char_tag(chars: str, tags: List[int]) -> List[int]:
    """
    make 32-bit numbers with mixing characters with output labels by bit shifting operation
    Args:
        chars:  list of characters
        tags:  list of output labels
    Returns:
        list of mixed numbers
    """
    char_nums = [ord(c) for c in chars]
    if tags[0] == SENT_DELIM_NUM:
        char_nums.insert(0, SENT_DELIM_NUM)
    if tags[-1] == SENT_DELIM_NUM:
        char_nums.append(SENT_DELIM_NUM)
    for idx, char_num in enumerate(char_nums):
        if char_num == ord(' '):
            char_nums[idx] = WORD_DELIM_NUM
            continue
        if tags[idx] == SENT_DELIM_NUM:
            continue
        char_nums[idx] = char_num << 12 | tags[idx]
    return char_nums
