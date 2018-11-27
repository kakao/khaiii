# -*- coding: utf-8 -*-


"""
형태소 분석 결과를 기술한 문자열을 파싱하는 모듈
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


#############
# constants #
#############
# 전체 태그셋. 숫자 -> 태그 매핑
TAGS = sorted(['EC', 'EF', 'EP', 'ETM', 'ETN', 'IC', 'JC', 'JKB', 'JKC', 'JKG',
               'JKO', 'JKQ', 'JKS', 'JKV', 'JX', 'MAG', 'MAJ', 'MM', 'NNB', 'NNG',
               'NNP', 'NP', 'NR', 'SE', 'SF', 'SH', 'SL', 'SN', 'SO', 'SP',
               'SS', 'SW', 'SWK', 'VA', 'VCN', 'VCP', 'VV', 'VX', 'XPN', 'XR',
               'XSA', 'XSN', 'XSV', 'ZN', 'ZV', 'ZZ', ])
# B- 태그가 가능한 태그 목록
B_TAGS = sorted(['EP', 'IC', 'JKB', 'JX', 'MAG', 'MM', 'NNB', 'NNG', 'NNP', 'NP',
                 'NR', 'SE', 'SF', 'SN', 'SO', 'SP', 'SS', 'SW', 'SWK', 'XPN',
                 'XR', 'XSN', ])
TAG_SET = {tag: num for num, tag in enumerate(TAGS, start=1)}    # 태그 -> 숫자 매핑

WORD_DELIM_STR = '_'    # 어절 경계(공백)를 나타내는 가상 형태소
SENT_DELIM_STR = '|'    # 문장 경계를 나타내는 가상 형태소
WORD_DELIM_NUM = -1    # 어절 경계 가상 태그 번호
SENT_DELIM_NUM = -2    # 문장 경계 가상 태그 번호


#########
# types #
#########
class ParseError(Exception):
    """
    형태소 분석 결과 문자열을 파싱하면서 발생하는 오류
    """
    pass


class Morph(object):
    """
    형태소
    """
    def __init__(self, lex, tag):
        """
        Arguments:
            lex:  형태소(어휘)
            tag:  품사 태그
        """
        self.lex = lex
        self.tag = tag

    def __str__(self):
        if not self.tag:
            return self.lex
        return '{}/{}'.format(self.lex, self.tag)

    def is_word_delim(self):
        """
        어절의 경계를 나타태는 지 여부
        Returns:
            어절의 경계 여부
        """
        return not self.tag and self.lex == WORD_DELIM_STR

    def is_sent_delim(self):
        """
        문장의 경계를 나타태는 지 여부
        Returns:
            문장의 경계 여부
        """
        return not self.tag and self.lex == SENT_DELIM_STR

    @classmethod
    def to_str(cls, morphs):
        """
        Morph 객체 리스트를 문자열로 변환한다.
        Arguments:
            morphs:  Morph 객체 리스트
        Returns:
            변환된 문자열
        """
        return ' + '.join([str(m) for m in morphs])

    @classmethod
    def parse(cls, morphs_str):
        """
        형태소 분석 결과 형태의 문자열을 파싱하여 Morph 객체 리스트를 반환하는 파싱 함수
        Arguments:
            morphs_str:  형태소 분석 결과 문자열. 예: "제이미/NNP + 는/JKS"
        Returns:
            Morph 객체 리스트
        """
        if not morphs_str:
            raise ParseError('empty to parse')
        return [cls._parse_one(m) for m in morphs_str.split(' + ')]

    @classmethod
    def _parse_one(cls, morph_str):
        """
        하나의 형태소 객체를 기술한 문자열을 파싱한다.
        Arguments:
            morph_str:  형태소 문자열
        Returns:
            Morph 객체
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
