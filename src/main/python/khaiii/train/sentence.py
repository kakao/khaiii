# -*- coding: utf-8 -*-


"""
sentence, word, morph, ...
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import logging
import re

from typing import List, Tuple


#########
# types #
#########
class Sentence:
    """
    raw sentence
    """
    def __init__(self, raw: str):
        """
        :param  raw:  raw sentence
        """
        self.words = raw.split()


class PosMorph:
    """
    morpheme
    """
    def __init__(self, morph: str, pos_tag: str = 'O', beg: int = -1, end: int = -1):
        self.morph = morph
        self.pos_tag = pos_tag
        self.beg = beg
        self.end = end

    def __str__(self):
        return '{}/{}'.format(self.morph, self.pos_tag)

    def __len__(self):
        return self.end - self.beg


class PosWord:
    """
    part-of-speech tagged word
    """
    def __init__(self, raw: str):
        """
        Args:
            raw:  raw word
        """
        self.raw = raw
        self.tags = []    # output tags for each character
        self.res_chrs = raw    # 원형 복원된 형태소들의 음절들을 합친 것
        self.res_tags = []    # 원형 복원된 형태소 음절들의 음절별 태그 (IOB)
        self.pos_tagged_morphs = [PosMorph(x, 'O', idx, idx+1) for idx, x in enumerate(raw)]

    def __str__(self):
        return '{}\t{}'.format(self.raw, ' '.join([str(x) for x in self.pos_tagged_morphs]))

    def for_pretrain(self) -> str:
        """
        pre-training을 위한 출력
        Returns:
            pre-training을 위한 문장
        """
        morph_strs = []
        morph = ''
        prev_tag = ''
        for char, iob_tag in zip(self.raw, self.tags):
            try:
                iob_tag, _ = iob_tag.split(':', 1)
            except ValueError:
                pass
            try:
                iob, tag = iob_tag.split('-')
            except ValueError as val_err:
                logging.error('raw: %s', self.raw)
                logging.error('tags: %s', self.tags)
                logging.error('iob_tag: %s', iob_tag)
                raise val_err
            if iob == 'B':
                if morph:
                    morph_strs.append('%s/%s' % (re.sub(r'\d', '0', morph), prev_tag))
                morph = char
                prev_tag = tag
            elif iob == 'I':
                if prev_tag == tag:
                    morph += char
                else:
                    if morph:
                        morph_strs.append('%s/%s' % (re.sub(r'\d', '0', morph), prev_tag))
                    morph = char
                    prev_tag = tag
        if morph:
            morph_strs.append('%s/%s' % (re.sub(r'\d', '0', morph), prev_tag))
        return ' '.join(morph_strs)

    def __eq__(self, other: 'PosWord'):
        """
        어절의 형태소 분석 결과가 일치할 경우 같다고 간주한다. (평가 프로그램에서 어절 단위 일치 여부 판단 시 사용)
        Args:
            other:  other object
        """
        return self.res_chrs == other.res_chrs and self.res_tags == other.res_tags

    def set_pos_result(self, tags: List[str], restore_dic: dict = None):
        """
        외부에서 생성된 PosWord객체의 정보를 현재 인스턴스에 설정합니다.
        Args:
            tags:  파일로 부터 읽은 형태소 태그(음절단위)
            restore_dic:  원형 복원 사전
        """
        if not restore_dic:
            tags = [x.split(':', 1)[0] for x in tags]
        self.tags = tags
        assert len(self.raw) == len(self.tags)    # 음절수와 태그수는 동일해야 한다.
        self.pos_tagged_morphs = self._make_pos_morphs(restore_dic)

    def _make_pos_morphs(self, restore_dic: dict = None):
        """
        형태소 태그리스트를 대상으로 B/I 로 병합되는 위치를 구합니다.
        Args:
            restore_dic:  원형 복원 사전
        """
        if not self.tags:
            return []

        self._restore(restore_dic)

        pos_morphs = []
        for beg, (lex, iob_tag) in enumerate(zip(self.res_chrs, self.res_tags)):
            try:
                iob, pos_tag = iob_tag.rsplit('-', 1)
            except ValueError as val_err:
                logging.error('invalid char/tag: %s/%s in [%s] %s', lex, iob_tag, self.res_chrs,
                              self.res_tags)
                raise val_err
            if iob == 'B' or not pos_morphs or pos_morphs[-1].pos_tag != pos_tag:
                pos_morphs.append(PosMorph(lex, pos_tag, beg, beg+1))
            elif iob == 'I':
                if pos_morphs[-1].pos_tag == pos_tag:
                    pos_morphs[-1].morph += lex
                    pos_morphs[-1].end += len(lex)
                else:
                    logging.debug('tag is different between B and I: %s vs %s',
                                  pos_morphs[-1].pos_tag, pos_tag)
                    pos_morphs.append(PosMorph(lex, pos_tag, beg, beg+1))
            else:
                raise ValueError('invalid IOB tag: {}/{} in [{}] {}'.format \
                                     (lex, iob_tag, self.res_chrs, self.res_tags))
        return pos_morphs

    def _restore(self, restore_dic: dict):
        """
        원형 복원 사전을 이용하여 형태소의 원형을 복원한다.
        Args:
            restore_dic:  원형 복원 사전
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
                else:
                    logging.debug('mapping not found: %s/%s', char, tag)
                    tag, _ = tag.split(':', 1)
            res_chrs.append(char)
            self.res_tags.append(tag)
        self.res_chrs = ''.join(res_chrs)


class PosSentence(Sentence):
    """
    part-of-speech tagged sentence
    """
    def __init__(self, raw: str):
        """
        Args:
            raw:  raw sentence
        """
        super().__init__(raw)
        self.pos_tagged_words = [] # list of PosWord

    def __str__(self):
        return '\n'.join([str(pos_word) for pos_word in self.pos_tagged_words])

    def get_beg_end_list(self) -> Tuple[List[int], List[int]]:
        """
        모든 형태소의 시작위치를 담는 리스트와, 끝 위치를 담는 리스트를 구합니다.
        Returns:
            list of begin positions
            list of end positions
        """
        begs = []
        ends = []
        for word in self.pos_tagged_words:
            for morph in word.pos_tagged_morphs:
                begs.append(morph.beg)
                ends.append(morph.end)
        return begs, ends

    def set_raw_by_words(self):
        """
        Sentence 객체의 'words' 멤버를 PosWords 객체의 raw를 이용하여 채운다.
        """
        self.words = [pos_word.raw for pos_word in self.pos_tagged_words]

    def init_pos_tags(self):
        """
        PosWord 객체를 생성하고 태그를 'O'로 세팅한다.
        """
        if self.pos_tagged_words:
            raise RuntimeError('PoS tagged words are already initialized')
        for word in self.words:
            self.pos_tagged_words.append(PosWord(word))

    def set_pos_result(self, tags: List[str], restore_dic: dict = None):
        """
        문장 전체에 대한 형태소 태그 출력 레이블 정보를 세팅하고 형태소를 복원한다.
        Args:
            tags:  문장 전체 태그 출력 레이블 정보
            restore_dic:  원형 복원 사전
        """
        total_char_num = 0
        for pos_word in self.pos_tagged_words:
            pos_word.set_pos_result(tags[total_char_num:total_char_num + len(pos_word.raw)],
                                    restore_dic)
            total_char_num += len(pos_word.raw)
        assert total_char_num == len(tags)

    def get_sequence(self, morph: bool = True, tag: bool = True, simple: bool = False) -> List[str]:
        """
        태그를 포함한 형태소 문자열을 생성하여 리턴합니다.
        Args:
            morph:  형태소 출력
            tag:  태그 표시 방법
            simple:  tag를 1byte만 출력
        Returns:
            문자열의 리스트
        """
        sequence = []
        for word in self.pos_tagged_words:
            for pos_morph in word.pos_tagged_morphs:
                morphs = []
                if morph:
                    morphs.append(pos_morph.morph)
                if tag:
                    morphs.append(pos_morph.pos_tag if not simple else pos_morph.pos_tag[0])
                sequence.append('/'.join(morphs))
        return sequence

    def get_all_morphs(self) -> List[str]:
        """
        문장을 구성하는 모든 PosMorph의 리스트를 리턴합니다.
        Returns:
            모든 형태소 리스트
        """
        return [morph for word in self.pos_tagged_words for morph in word.pos_tagged_morphs]
