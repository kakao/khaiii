# -*- coding: utf-8 -*-


"""
형태소 분석 결과와 원문의 음절을 정렬하는 모듈
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import codecs
from collections import Counter, defaultdict
import logging
import os

import jaso


#########
# types #
#########
class Aligner(object):
    """
    음절과 형태소 분석 결과의 정렬을 수행하는 클래스
    """
    def __init__(self, rsc_src):
        """
        리소스를 오픈하고 초기화한다.
        Args:
            rsc_src:  training 리소스 디렉토리
        """
        self.align_map = {}    # M:N mapping dictionary
        # middle mapping candidates examples after forward/backward prefix/suffix mapping
        self.middle_unmapped = defaultdict(Counter)
        self._open(rsc_src)

    def align(self, word):
        """
        어절의 원문과 분석 결과를 음절 단위로 정렬(매핑)한다.
        Args:
            word:  어절 객체
        Returns:
            매핑 정보. 음절의 갯수 만큼의 매핑되는 형태소 분석 결과 음절의 리스트
        """
        mrp_chrs = []
        for idx, morph in enumerate(word.morphs):
            for jdx, morph_char in enumerate(morph.lex):
                iob = 'I'
                if jdx == 0 and idx > 0 and word.morphs[idx-1].tag == morph.tag:
                    # 이전 형태소와 현재 형태소의 품사가 같을 경우에만 B- 태그를 갖는다. (IOB1 방식)
                    iob = 'B'
                mrp_chrs.append(MrpChr(morph_char, '{}-{}'.format(iob, morph.tag)))
        raw_morph = self._get_morph_raw(word)
        if word.raw == raw_morph:
            return [[c, ] for c in mrp_chrs]
        word_norm = self._norm(word.raw)
        morph_norm = self._norm(raw_morph)
        if word_norm == morph_norm:
            return self._align_phoneme(word.raw, mrp_chrs)
        try:
            return self._align_forward_backward(word.raw, mrp_chrs)
        except IndexError as idx_err:
            raise AlignError(idx_err)

    def print_middle_cnt(self):
        """
        print middle mapping counts
        """
        cmp = lambda item: sum(item[1].values())
        for (word, mrp_chrs_str), exmpl_cnt in sorted(self.middle_unmapped.items(), key=cmp,
                                                      reverse=True):
            logging.debug('[%d] %s => %s', sum(exmpl_cnt.values()), word, mrp_chrs_str)
            for exmpl, _ in exmpl_cnt.most_common(10):
                logging.debug('    %s', exmpl)
        logging.info('total number of unmapped pairs: %d',
                     sum([sum(cnt.values()) for cnt in self.middle_unmapped.values()]))

    def _open(self, rsc_dir):
        """
        initialize resources
        Args:
            rsc_dir:  resource dir
        """
        file_path = '{rsc_dir}/char_align.map'.format(**locals())
        file_name = os.path.basename(file_path)
        for line_num, line in enumerate(codecs.open(file_path, 'r', encoding='UTF-8'), start=1):
            line = line.rstrip('\r\n')
            if not line or line[0] == '#':
                continue
            try:
                raw_word, mrp_chrs_str, map_nums = line.split('\t')
            except ValueError as val_err:
                raise ValueError('{}({}): {}: {}'.format(file_name, line_num, val_err, line))
            key = raw_word, mrp_chrs_str
            if key in self.align_map:
                raise ValueError('{}({}): duplicated M:N entry: {} => {}: {} vs {}'.format(
                    file_name, line_num, raw_word, mrp_chrs_str, self.align_map[key], map_nums))
            if len(raw_word) != len(map_nums) or \
                    len(mrp_chrs_str.split()) != sum([int(n) for n in map_nums]):
                raise ValueError('{}({}): invalid M:N dic entry: {} =({})=> {}'.format(
                    file_name, line_num, raw_word, map_nums, mrp_chrs_str))
            self.align_map[key] = map_nums

    @classmethod
    def _get_morph_raw(cls, word):
        """
        get raw string from morphemes
        Args:
            word:  eojeol
        Returns:
            raw string from morpheme
        """
        return ''.join([m.lex for m in word.morphs])

    @classmethod
    def _norm(cls, text):
        """
        unicode normalization of text
        Args:
            text:  text
        Returns:
            normalized text
        """
        return jaso.decompose(text)

    @classmethod
    def _align_phoneme(cls, raw_word, mrp_chrs):
        """
        align word with morpheme which is same phoneme
        Args:
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            tag list
        """
        maps = []
        mrp_chrs_idx = 0    # index for mrp_chrs
        for word_char in raw_word:
            morph_char = mrp_chrs[mrp_chrs_idx].char
            if word_char == morph_char:
                maps.append([mrp_chrs[mrp_chrs_idx], ])
            else:
                word_norm = cls._norm(word_char)
                char_norm = cls._norm(morph_char)
                if word_norm == char_norm:
                    maps.append([mrp_chrs[mrp_chrs_idx], ])
                elif mrp_chrs_idx+1 < len(mrp_chrs) and \
                        word_norm == (char_norm + mrp_chrs[mrp_chrs_idx+1].char):
                    maps.append(mrp_chrs[mrp_chrs_idx:mrp_chrs_idx+2])    # 1:2 mapping
                    mrp_chrs_idx += 1
                elif (word_char == '㈜' and mrp_chrs_idx+2 < len(mrp_chrs) and morph_char == '('
                      and mrp_chrs[mrp_chrs_idx+1].char == '주'
                      and mrp_chrs[mrp_chrs_idx+2].char == ')'):
                    maps.append(mrp_chrs[mrp_chrs_idx:mrp_chrs_idx+3])    # 1:3 mapping
                    mrp_chrs_idx += 2
                else:
                    logging.info('fail to map word: %s', str(raw_word))
                    return []
            mrp_chrs_idx += 1
        return maps

    @classmethod
    def _align_forward(cls, raw_word, mrp_chrs):
        """
        align from front of word
        Args:
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            word index
            mrp_chrs index
        """
        word_idx = 0
        mrp_chrs_idx = 0
        for word_char in raw_word:
            morph_char = mrp_chrs[mrp_chrs_idx].char
            if word_char != morph_char:
                word_norm = cls._norm(word_char)
                char_norm = cls._norm(morph_char)
                if word_norm == char_norm:
                    pass
                elif mrp_chrs_idx+1 < len(mrp_chrs) and \
                        word_norm == cls._norm(morph_char + mrp_chrs[mrp_chrs_idx + 1].char):
                    mrp_chrs_idx += 1
                else:
                    return word_idx, mrp_chrs_idx
            word_idx += 1
            mrp_chrs_idx += 1
        return word_idx, mrp_chrs_idx

    @classmethod
    def _align_backward(cls, raw_word, mrp_chrs):
        """
        align from back of word
        Args:
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            word index
            mrp_chrs index
        """
        word_idx = len(raw_word) - 1
        mrp_chrs_idx = len(mrp_chrs) - 1
        while word_idx >= 0:
            word_char = raw_word[word_idx]
            morph_char = mrp_chrs[mrp_chrs_idx].char
            if word_char != morph_char:
                word_norm = cls._norm(word_char)
                char_norm = cls._norm(morph_char)
                if word_norm == char_norm:
                    pass
                elif mrp_chrs_idx+1 >= 0 and \
                        word_norm == cls._norm(mrp_chrs[mrp_chrs_idx - 1].char + morph_char):
                    mrp_chrs_idx -= 1
                elif (word_char == '㈜' and mrp_chrs_idx-2 >= 0 and morph_char == ')'
                      and mrp_chrs[mrp_chrs_idx-1].char == '주'
                      and mrp_chrs[mrp_chrs_idx-2].char == '('):
                    mrp_chrs_idx -= 2
                else:
                    return word_idx+1, mrp_chrs_idx+1
            word_idx -= 1
            mrp_chrs_idx -= 1
        return word_idx+1, mrp_chrs_idx+1

    @classmethod
    def _is_verb_ending(cls, verb, ending):
        """
        whether is verb + ending pattern or not
        Args:
            verb:  verb part. (lex, tag) pair
            ending:  ending part. (lex, tag) pair
        Returns:
            whether is verb + ending
        """
        verb_tag = verb.tag[2:]
        ending_tag = ending.tag[2:]
        return verb_tag in {'VV', 'VA', 'VCP', 'VCN', 'VX', 'XSV', 'XSA', 'EP'} and \
            ending_tag in {'EC', 'EP', 'EF', 'ETN', 'ETM'}

    @classmethod
    def _are_first_last_phoneme_same(cls, raw_word, mrp_chrs):
        """
        whether are same the first phoneme and last phoneme
        Args:
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            whether is same or not
        """
        word_norm = cls._norm(raw_word)
        morph_norm = cls._norm(''.join([_.char for _ in mrp_chrs]))
        return word_norm[0] == morph_norm[0] and word_norm[-1] == morph_norm[-1]

    @classmethod
    def _is_ah_ending_verb(cls, mrp_chr):
        """
        whether 'ㅏ' ending verb or not
        Args:
            mrp_chr:  MrpChr object
        Returns:
            whether 'ㅏ' ending verb or not
        """
        if mrp_chr.tag[2:] not in ['VV', 'VA', 'VX']:
            return False
        norm_char = cls._norm(mrp_chr.char)
        return len(norm_char) == 2 and norm_char[1] == 'ᅡ'    # code is 4449, not 12623

    @classmethod
    def _is_eo_ending_verb(cls, mrp_chr):
        """
        whether 'ㅓ', 'ㅐ' ending verb or not
        Args:
            mrp_chr:  MrpChr object
        Returns:
            whether 'ㅓ', 'ㅐ' ending verb or not
        """
        if mrp_chr.tag[2:] not in ['VV', 'VA', 'VX']:
            return False
        norm_char = cls._norm(mrp_chr.char)
        return (len(norm_char) == 2 and
                norm_char[1] in ['ᅥ', 'ᅧ', 'ᅢ'])    # code is 4453, 4455, 4450

    @classmethod
    def _align_middle_zero2one(cls, pfx_word, pfx_map, mdl_mrp_chrs, sfx_word, sfx_map):
        """
        align middle chunks after forward/backward aligning which has no middle raw character,
            but has a remaining middle single morpheme character
        Args:
            pfx_word:  word prefix
            pfx_map:  prefix mapping
            mdl_mrp_chrs:  remaining middle morpheme character
            sfx_word:  word suffix
            sfx_map:  suffix mapping
        """
        def _attach_to_sfx():
            # attach to the first part of suffix
            logging.debug('[0:1] %s --> %s', MrpChr.to_str(mdl_mrp_chrs), sfx_word)
            sfx_map[0].insert(0, mrp_chr)
            del mdl_mrp_chrs[:]

        def _attach_to_pfx():
            # attach to the last part of prefix
            logging.debug('[0:1] %s <-- %s', pfx_word, MrpChr.to_str(mdl_mrp_chrs))
            pfx_map[-1].append(mrp_chr)
            del mdl_mrp_chrs[:]

        mrp_chr = mdl_mrp_chrs[0]
        if not pfx_map and sfx_map:
            _attach_to_sfx()
        elif pfx_map and not sfx_map:
            _attach_to_pfx()
        elif sfx_map and (mrp_chr.tag.startswith('B-') or sfx_map[0][0].tag[2:] == mrp_chr.tag[2:]):
            _attach_to_sfx()
        elif pfx_map and pfx_map[-1][-1].tag[2:] != mrp_chr.tag[2:]:
            _attach_to_sfx()
        elif pfx_map:
            _attach_to_pfx()
        else:
            raise RuntimeError('nowhere attach to')

    @classmethod
    def _is_share_phoneme(cls, mdl_word, mdl_mrp_chrs):
        """
        whether middle word characters and morpheme characters share same phoneme
        Args:
            mdl_word:  middle word chunk
            mdl_mrp_chrs:  middle MrpChr object sequence
        Returns:
            whether share or not
        """
        if len(mdl_word) == 1:
            # whether share first two phonemes for single character
            return cls._norm(mdl_word)[0:2] == cls._norm(mdl_mrp_chrs[0].char)[0:2]
        for word_char, mrp_chr in zip(mdl_word, mdl_mrp_chrs):
            # all characters share first phoneme
            phonemes = cls._norm(word_char)
            first_phoneme = phonemes[1] if phonemes[0] == 'ᄋ' else phonemes[0]
            if first_phoneme not in cls._norm(mrp_chr.char):
                return False
        return True

    def _align_middle_by_dic(self, mdl_word, mdl_mrp_chrs):
        """
        align middle chunks after forward/backward aligning with mapping dictionary
        Args:
            mdl_word:  middle word chunk
            mdl_mrp_chrs:  middle MrpChr object sequence
        Returns:
            aligned list of tags. empty list if not found in dictionary
        """
        maps = []
        mdl_mrp_chrs_str = MrpChr.to_str(mdl_mrp_chrs)
        dic_key = mdl_word, mdl_mrp_chrs_str
        if dic_key in self.align_map:
            # M:N mapping by dictionary
            map_nums = self.align_map[dic_key]
            logging.debug('[M:N] %s =(%s)=> %s', mdl_word, map_nums, mdl_mrp_chrs_str)
            idx = 0
            for map_num in map_nums:
                maps.append(mdl_mrp_chrs[idx:idx + int(map_num)])
                idx += int(map_num)
        return maps

    def _align_middle(self, mdl_word, mdl_mrp_chrs, raw_word, mrp_chrs):
        """
        align middle chunks after forward/backward aligning
        Args:
            mdl_word:  middle word chunk
            mdl_mrp_chrs:  middle MrpChr object sequence
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            aligned list of tags
        """
        maps = []
        mdl_mrp_chrs_str = MrpChr.to_str(mdl_mrp_chrs)
        if len(mdl_word) == 1 and len(mdl_mrp_chrs) == 2 and \
                self._is_verb_ending(mdl_mrp_chrs[0], mdl_mrp_chrs[1]):
            # 1:2 mapping. (용언+어미) 패턴
            logging.debug('[1:2] %s => %s', mdl_word, mdl_mrp_chrs_str)
            maps.append(mdl_mrp_chrs)
        elif len(mdl_word) == 2 and len(mdl_mrp_chrs) == 4 and \
                self._is_verb_ending(mdl_mrp_chrs[0], mdl_mrp_chrs[1]) and \
                self._is_verb_ending(mdl_mrp_chrs[2], mdl_mrp_chrs[3]):
            # 1:2 + 1:2 mapping. (용언+어미+용언+어미) 패턴
            logging.debug('[1:2+1:2] %s => %s', mdl_word, mdl_mrp_chrs_str)
            maps.extend([mdl_mrp_chrs[0:2], mdl_mrp_chrs[2:4]])
        elif len(mdl_word) == 1 and len(mdl_mrp_chrs) == 1 and \
                mdl_mrp_chrs[0].tag[2:] in {'VV', 'VA', 'VX'}:
            # 1:1 mapping. (용언) 패턴
            logging.debug('[1:1] %s => %s', mdl_word, mdl_mrp_chrs_str)
            maps.append(mdl_mrp_chrs)
        elif len(mdl_word) == 2 and len(mdl_mrp_chrs) == 2 and \
                self._are_first_last_phoneme_same(mdl_word, mdl_mrp_chrs):
            # 1:1 + 1:1 mapping
            logging.debug('[1:1+1:1] %s => %s', mdl_word, mdl_mrp_chrs_str)
            maps.extend([mdl_mrp_chrs[0:1], mdl_mrp_chrs[1:2]])
        elif len(mdl_word) == 2 and len(mdl_mrp_chrs) == 3 and \
                mdl_mrp_chrs[0].tag[2:] == mdl_mrp_chrs[1].tag[2:] and \
                self._is_verb_ending(mdl_mrp_chrs[1], mdl_mrp_chrs[2]):
            # 1:2 + 1:1 mapping. (용언+어미 패턴)
            logging.debug('[1:2+1:1] %s => %s', mdl_word, mdl_mrp_chrs_str)
            maps.extend([mdl_mrp_chrs[0:2], mdl_mrp_chrs[2:3]])
        elif len(mdl_word) == len(mdl_mrp_chrs) and self._is_share_phoneme(mdl_word, mdl_mrp_chrs):
            # N:N mapping. but they look like errors! :(
            logging.debug('[N:N] %s => %s', mdl_word, mdl_mrp_chrs_str)
            maps.extend([[_, ] for _ in mdl_mrp_chrs])
        else:
            # mapping by dictionary
            maps.extend(self._align_middle_by_dic(mdl_word, mdl_mrp_chrs))
        if not maps:
            example = raw_word, MrpChr.to_str(mrp_chrs)
            self.middle_unmapped[mdl_word, mdl_mrp_chrs_str][example] += 1
        return maps

    @classmethod
    def _align_middle_preproc(cls, pfx_mrp_chrs, pfx_map, mdl_mrp_chrs, sfx_mrp_chrs, sfx_map):
        """
        pre-processing middle part after forward/backward mapping before applying rules
        Args:
            pfx_mrp_chrs:  prefix list of MrpChr object
            pfx_map:  mapping of prefix
            mdl_mrp_chrs:  middle list of MrpChr object
            sfx_mrp_chrs:  suffix list of MrpChr object
            sfx_map:  mapping of suffix
        """
        if pfx_mrp_chrs and mdl_mrp_chrs and mdl_mrp_chrs[0] == MrpChr('아', 'I-EC') and \
                cls._is_ah_ending_verb(pfx_mrp_chrs[-1]):
            # check the first '아/I-EC' in middle whether can be attached to end of prefix
            pfx_mrp_chrs.append(mdl_mrp_chrs[0])
            pfx_map[-1].append(mdl_mrp_chrs[0])
            del mdl_mrp_chrs[0]
        elif pfx_mrp_chrs and mdl_mrp_chrs and mdl_mrp_chrs[0] == MrpChr('어', 'I-EC') and \
                cls._is_eo_ending_verb(pfx_mrp_chrs[-1]):
            # check the first '어/I-EC' in middle whether can be attached to end of prefix
            pfx_mrp_chrs.append(mdl_mrp_chrs[0])
            pfx_map[-1].append(mdl_mrp_chrs[0])
            del mdl_mrp_chrs[0]
        if mdl_mrp_chrs and sfx_mrp_chrs and mdl_mrp_chrs[-1] == MrpChr('이', 'I-VCP') \
                and sfx_mrp_chrs[0].tag in ['I-EC', 'I-EF', 'I-EP', 'I-ETM', 'I-ETN']:
            # check the last '이/I-VCP' in middle whether can be attached to first of suffix
            sfx_mrp_chrs.insert(0, mdl_mrp_chrs[-1])
            sfx_map[0].insert(0, mdl_mrp_chrs[-1])
            del mdl_mrp_chrs[-1]
        elif mdl_mrp_chrs and sfx_mrp_chrs and mdl_mrp_chrs[-1] == MrpChr('하', 'I-VX') \
                and sfx_mrp_chrs[0] == MrpChr('겠', 'I-EP'):
            # check the last '하/I-VX' in middle whether can be attached to first of suffix
            if (len(mdl_mrp_chrs) > 1 and mdl_mrp_chrs[-2] == MrpChr('야', 'I-EC')) or \
                    (pfx_mrp_chrs and pfx_mrp_chrs[-1] == MrpChr('야', 'I-EC')):
                sfx_mrp_chrs.insert(0, mdl_mrp_chrs[-1])
                sfx_map[0].insert(0, mdl_mrp_chrs[-1])
                del mdl_mrp_chrs[-1]

    def _get_pfx_mdl_sfx(self, raw_word, mrp_chrs):
        """
        get prefix, middle, suffix after forward/backward align
        Args:
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            many many tuples..
        """
        word_pfx_idx, mrp_chrs_pfx_idx = self._align_forward(raw_word, mrp_chrs)
        pfx_word = raw_word[:word_pfx_idx]
        pfx_mrp_chrs = mrp_chrs[:mrp_chrs_pfx_idx]
        word_sfx_idx, mrp_chrs_sfx_idx = self._align_backward(raw_word[word_pfx_idx:],
                                                              mrp_chrs[mrp_chrs_pfx_idx:])
        word_sfx_idx += word_pfx_idx
        mrp_chrs_sfx_idx += mrp_chrs_pfx_idx
        sfx_word = raw_word[word_sfx_idx:]
        sfx_mrp_chrs = mrp_chrs[mrp_chrs_sfx_idx:]
        mdl_word = raw_word[word_pfx_idx:word_sfx_idx]
        mdl_mrp_chrs = mrp_chrs[mrp_chrs_pfx_idx:mrp_chrs_sfx_idx]
        pfx_map = self._align_phoneme(pfx_word, pfx_mrp_chrs)
        sfx_map = self._align_phoneme(sfx_word, sfx_mrp_chrs)
        self._align_middle_preproc(pfx_mrp_chrs, pfx_map, mdl_mrp_chrs, sfx_mrp_chrs, sfx_map)
        return (pfx_word, mdl_word, sfx_word), \
               (pfx_mrp_chrs, mdl_mrp_chrs, sfx_mrp_chrs), \
               (pfx_map, sfx_map)

    def _align_forward_backward(self, raw_word, mrp_chrs):
        """
        align word with morpheme which is same phoneme
        Args:
            raw_word:  raw word
            mrp_chrs:  list of MrpChr object
        Returns:
            tag list
        """
        # align forward/backward and get pieces of alignment
        (pfx_word, mdl_word, sfx_word), \
            (pfx_mrp_chrs, mdl_mrp_chrs, sfx_mrp_chrs), \
            (pfx_map, sfx_map) = self._get_pfx_mdl_sfx(raw_word, mrp_chrs)

        if not mdl_word and not mdl_mrp_chrs:
            return pfx_map + sfx_map
        elif not mdl_word:
            if len(mdl_mrp_chrs) == 1:
                self._align_middle_zero2one(pfx_word, pfx_map, mdl_mrp_chrs, sfx_word, sfx_map)
                return pfx_map + sfx_map
            algn_err = AlignError('{0:N}')
            algn_err.add_msg('[%s] [%s] [%s]' % (pfx_word, mdl_word, sfx_word))
            algn_err.add_msg('[%s] [%s] [%s]' % \
                             (MrpChr.to_str(pfx_mrp_chrs), MrpChr.to_str(mdl_mrp_chrs),
                              MrpChr.to_str(sfx_mrp_chrs)))
            raise algn_err
        elif not mdl_mrp_chrs:
            algn_err = AlignError('{N:0}')
            algn_err.add_msg('[%s] [%s] [%s]' % (pfx_word, mdl_word, sfx_word))
            algn_err.add_msg('[%s] [%s] [%s]' % \
                             (MrpChr.to_str(pfx_mrp_chrs), MrpChr.to_str(mdl_mrp_chrs),
                              MrpChr.to_str(sfx_mrp_chrs)))
            raise algn_err
        mdl_map = self._align_middle(mdl_word, mdl_mrp_chrs, raw_word, mrp_chrs)
        if not mdl_map:
            algn_err = AlignError('{M:N}')
            algn_err.add_msg('[%s] [%s] [%s]' % (pfx_word, mdl_word, sfx_word))
            algn_err.add_msg('[%s] [%s] [%s]' % \
                             (MrpChr.to_str(pfx_mrp_chrs), MrpChr.to_str(mdl_mrp_chrs),
                              MrpChr.to_str(sfx_mrp_chrs)))
            raise algn_err
        return pfx_map + mdl_map + sfx_map


class AlignError(Exception):
    """
    음절 정렬 과정에서 나타나는 예외
    """
    def __init__(self, pfx):
        """
        Args:
            pfx:  예외 출력 시 보여줄 prefix (카테고리)
        """
        super().__init__()
        self._pfx = pfx
        self._msgs = []

    def __str__(self):
        return '\n'.join(['%s %s' % (self._pfx, _) for _ in self._msgs] + ['', ])

    def add_msg(self, msg):
        """
        메세제를 추가한다.
        Args:
            msg:  에러 메세지
        """
        self._msgs.append(msg)


class MrpChr(object):    # pylint: disable=too-few-public-methods
    """
    음절과 태그 pair
    """
    def __init__(self, char, tag):
        """
        Args:
            char:  음절
            tag:  태그
        """
        self.char = char
        self.tag = tag

    def __str__(self):
        return '%s/%s' % (self.char, self.tag)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        """
        Args:
            other:  다른 객체
        Returns:
            같을 경우 True
        """
        return self.char == other.char and self.tag == other.tag

    @classmethod
    def to_str(cls, mrp_chrs):
        """
        MrpChr 객체 리스트를 문자열로 변환하는 메소드
        Args:
            mrp_chrs:  MrpChr 객체 리스트
        Returns:
            변환된 문자열
        """
        return ' '.join([str(_) for _ in mrp_chrs])
