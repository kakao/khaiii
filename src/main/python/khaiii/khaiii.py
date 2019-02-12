#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
khaiii API module
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import ctypes
from ctypes.util import find_library
import logging
import os
import platform
import sys
from typing import List


#########
# types #
#########
class _khaiii_morph_t(ctypes.Structure):    # pylint: disable=invalid-name,too-few-public-methods
    """
    khaiii_morph_t structure
    """


_khaiii_morph_t._fields_ = [    # pylint: disable=protected-access
    ('lex', ctypes.c_char_p),
    ('tag', ctypes.c_char_p),
    ('begin', ctypes.c_int),
    ('length', ctypes.c_int),
    ('reserved', ctypes.c_char * 8),
    ('next', ctypes.POINTER(_khaiii_morph_t)),
]


class _khaiii_word_t(ctypes.Structure):    # pylint: disable=invalid-name,too-few-public-methods
    """
    khaiii_word_t structure
    """


_khaiii_word_t._fields_ = [    # pylint: disable=protected-access
    ('begin', ctypes.c_int),
    ('length', ctypes.c_int),
    ('reserved', ctypes.c_char * 8),
    ('morphs', ctypes.POINTER(_khaiii_morph_t)),
    ('next', ctypes.POINTER(_khaiii_word_t)),
]


class KhaiiiExcept(Exception):
    """
    khaiii API를 위한 표준 예외 클래스
    """


class KhaiiiMorph:
    """
    형태소 객체
    """
    def __init__(self):
        self.lex = ''
        self.tag = ''
        self.begin = -1    # 음절 시작 위치
        self.length = -1    # 음절 길이
        self.reserved = b''

    def __str__(self):
        return '{}/{}'.format(self.lex, self.tag)

    def set(self, morph: ctypes.POINTER(_khaiii_morph_t), align: List[List[int]]):
        """
        khaiii_morph_t 구조체로부터 형태소 객체의 내용을 채운다.
        Args:
            morph:  khaiii_morph_t 구조체 포인터
            align:  byte-음절 정렬 정보
        """
        assert morph.contents
        self.lex = morph.contents.lex.decode('UTF-8')
        self.begin = align[morph.contents.begin]
        end = align[morph.contents.begin + morph.contents.length - 1] + 1
        self.length = end - self.begin
        if morph.contents.tag:
            self.tag = morph.contents.tag.decode('UTF-8')
        self.reserved = morph.contents.reserved


class KhaiiiWord:
    """
    어절 객체
    """
    def __init__(self):
        self.lex = ''
        self.begin = -1    # 음절 시작 위치
        self.length = -1    # 음절 길이
        self.reserved = b''
        self.morphs = []

    def __str__(self):
        morphs_str = ' + '.join([str(m) for m in self.morphs])
        return '{}\t{}'.format(self.lex, morphs_str)

    def set(self, word: ctypes.POINTER(_khaiii_word_t), in_str: str, align: list):
        """
        khaiii_word_t 구조체로부터 어절 객체의 내용을 채운다.
        Args:
            word:  khaiii_word_t 구조체 포인터
            in_str:  입력 문자열
            align:  byte-음절 정렬 정보
        """
        assert word.contents
        self.begin = align[word.contents.begin]
        end = align[word.contents.begin + word.contents.length - 1] + 1
        self.length = end - self.begin
        self.lex = in_str[self.begin:end]
        self.reserved = word.contents.reserved
        self.morphs = self._make_morphs(word.contents.morphs, align)

    @classmethod
    def _make_morphs(cls, morph_head: ctypes.POINTER(_khaiii_morph_t), align: list) \
            -> List[KhaiiiMorph]:
        """
        어절 내에 포함된 형태소의 리스트를 생성한다.
        Args:
            morph_head:  linked-list 형태의 형태소 헤드
            align:  byte-음절 정렬 정보
        Returns:
            형태소 객체 리스트
        """
        morphs = []
        ptr = morph_head
        while ptr:
            morph = KhaiiiMorph()
            morph.set(ptr, align)
            morphs.append(morph)
            ptr = ptr.contents.next
        return morphs


class KhaiiiApi:
    """
    khaiii API 객체
    """
    def __init__(self, lib_path: str = '', rsc_dir: str = '', opt_str: str = '',
                 log_level: str = 'warn'):
        """
        Args:
            lib_path:  (shared) 라이브러리의 경로
            rsc_dir:  리소스 디렉토리
            opt_str:  옵션 문자열 (JSON 포맷)
            log_level:  로그 레벨 (trace, debug, info, warn, err, critical)
        """
        self._handle = -1
        if not lib_path:
            lib_name = 'libkhaiii.dylib' if platform.system() == 'Darwin' else 'libkhaiii.so'
            lib_dir = os.path.join(os.path.dirname(__file__), 'lib')
            lib_path = '{}/{}'.format(lib_dir, lib_name)
            if not os.path.exists(lib_path):
                lib_path = find_library(lib_name)
                if not lib_path:
                    logging.error('current working directory: %s', os.getcwd())
                    logging.error('library directory: %s', lib_dir)
                    raise KhaiiiExcept('fail to find library: {}'.format(lib_name))
        logging.debug('khaiii library path: %s', lib_path)
        self._lib = ctypes.CDLL(lib_path)
        self._set_arg_res_types()
        self.set_log_level('all', log_level)
        self.open(rsc_dir, opt_str)

    def __del__(self):
        self.close()

    def version(self) -> str:
        """
        khaiii_version() API
        Returns:
            버전 문자열
        """
        return self._lib.khaiii_version().decode('UTF-8')

    def open(self, rsc_dir: str = '', opt_str: str = ''):
        """
        khaiii_open() API
        Args:
            rsc_dir:  리소스 디렉토리
            opt_str:  옵션 문자열 (JSON 포맷)
        """
        self.close()
        if not rsc_dir:
            rsc_dir = os.path.join(os.path.dirname(__file__), 'share/khaiii')
        self._handle = self._lib.khaiii_open(rsc_dir.encode('UTF-8'), opt_str.encode('UTF-8'))
        if self._handle < 0:
            raise KhaiiiExcept(self._last_error())
        logging.info('khaiii opened with rsc_dir: "%s", opt_str: "%s"', rsc_dir, opt_str)

    def close(self):
        """
        khaiii_close() API
        """
        if self._handle >= 0:
            self._lib.khaiii_close(self._handle)
            logging.debug('khaiii closed')
        self._handle = -1

    def analyze(self, in_str: str, opt_str: str = '') -> List[KhaiiiWord]:
        """
        khaiii_analyze() API
        Args:
            in_str:  입력 문자열
            opt_str:  동적 옵션 (JSON 포맷)
        Returns:
            분셕 결과. 어절(KhaiiiWord) 객체의 리스트
        """
        assert self._handle >= 0
        results = self._lib.khaiii_analyze(self._handle, in_str.encode('UTF-8'),
                                           opt_str.encode('UTF-8'))
        if not results:
            raise KhaiiiExcept(self._last_error())
        words = self._make_words(in_str, results)
        self._free_results(results)
        return words

    def analyze_bfr_errpatch(self, in_str: str, opt_str: str = '') -> List[int]:
        """
        khaiii_analyze_bfr_errpatch() dev API
        Args:
            in_str:  입력 문자열
            opt_str:  동적 옵션 (JSON 포맷)
        Returns:
            음절별 태그 값의 리스트
        """
        assert self._handle >= 0
        in_bytes = in_str.encode('UTF-8')
        output = (ctypes.c_short * (len(in_bytes) + 3))()
        ctypes.cast(output, ctypes.POINTER(ctypes.c_short))
        out_num = self._lib.khaiii_analyze_bfr_errpatch(self._handle, in_bytes,
                                                        opt_str.encode('UTF-8'),
                                                        output)
        if out_num < 2:
            raise KhaiiiExcept(self._last_error())
        results = []
        for idx in range(out_num):
            results.append(output[idx])
        return results

    def set_log_level(self, name: str, level: str):
        """
        khaiii_set_log_level() dev API
        Args:
            name:  로거 이름
            level:  로거 레벨. trace, debug, info, warn, err, critical
        """
        ret = self._lib.khaiii_set_log_level(name.encode('UTF-8'), level.encode('UTF-8'))
        if ret < 0:
            raise KhaiiiExcept(self._last_error())

    def set_log_levels(self, name_level_pairs: str):
        """
        khaiii_set_log_levels() dev API
        Args:
            name_level_pairs:  로거 (이름, 레벨) 쌍의 리스트.
                               "all:warn,console:info,Tagger:debug"와 같은 형식
        """
        ret = self._lib.khaiii_set_log_levels(name_level_pairs.encode('UTF-8'))
        if ret < 0:
            raise KhaiiiExcept(self._last_error())

    def _free_results(self, results: ctypes.POINTER(_khaiii_word_t)):
        """
        khaiii_free_results() API
        Args:
            results:  analyze() 메소드로부터 받은 분석 결과
        """
        assert self._handle >= 0
        self._lib.khaiii_free_results(self._handle, results)

    def _last_error(self) -> str:
        """
        khaiii_last_error() API
        Returns:
            오류 메세지
        """
        return self._lib.khaiii_last_error(self._handle).decode('UTF-8')

    def _set_arg_res_types(self):
        """
        라이브러리 함수들의 argument 타입과 리턴 타입을 지정
        """
        self._lib.khaiii_version.argtypes = None
        self._lib.khaiii_version.restype = ctypes.c_char_p
        self._lib.khaiii_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.khaiii_open.restype = ctypes.c_int
        self._lib.khaiii_close.argtypes = [ctypes.c_int, ]
        self._lib.khaiii_close.restype = None
        self._lib.khaiii_analyze.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self._lib.khaiii_analyze.restype = ctypes.POINTER(_khaiii_word_t)
        self._lib.khaiii_free_results.argtypes = [ctypes.c_int, ctypes.POINTER(_khaiii_word_t)]
        self._lib.khaiii_free_results.restype = None
        self._lib.khaiii_last_error.argtypes = [ctypes.c_int, ]
        self._lib.khaiii_last_error.restype = ctypes.c_char_p
        self._lib.khaiii_analyze_bfr_errpatch.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                                          ctypes.c_char_p,
                                                          ctypes.POINTER(ctypes.c_short)]
        self._lib.khaiii_analyze_bfr_errpatch.restype = ctypes.c_int
        self._lib.khaiii_set_log_level.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.khaiii_set_log_level.restype = ctypes.c_int
        self._lib.khaiii_set_log_levels.argtypes = [ctypes.c_char_p, ]
        self._lib.khaiii_set_log_levels.restype = ctypes.c_int

    @classmethod
    def _make_words(cls, in_str: str, results: ctypes.POINTER(_khaiii_word_t)) -> List[KhaiiiWord]:
        """
        linked-list 형태의 API 분석 결과로부터 어절(KhaiiiWord) 객체의 리스트를 생성
        Args:
            in_str:  입력 문자열
            results:  분석 결과
        Returns:
            어절(KhaiiiWord) 객체의 리스트
        """
        align = cls._get_align(in_str)
        words = []
        ptr = results
        while ptr:
            word = KhaiiiWord()
            word.set(ptr, in_str, align)
            words.append(word)
            ptr = ptr.contents.next
        return words

    @classmethod
    def _get_align(cls, in_str: str) -> List[List[int]]:
        """
        byte-음절 정렬 정보를 생성. byte 길이 만큼의 각 byte 위치별 음절 위치
        Args:
            in_str:  입력 문자열
        Returns:
            byte-음절 정렬 정보
        """
        align = []
        for idx, char in enumerate(in_str):
            utf8 = char.encode('UTF-8')
            align.extend([idx, ] * len(utf8))
        return align


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    khaiii_api = KhaiiiApi(args.lib_path, args.rsc_dir, args.opt_str)
    if args.set_log:
        khaiii_api.set_log_levels(args.set_log)
    for line in sys.stdin:
        if args.errpatch:
            print(khaiii_api.analyze_bfr_errpatch(line, ''))
            continue
        words = khaiii_api.analyze(line, '')
        for word in words:
            print(word)
        print()


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='khaiii API module test program')
    parser.add_argument('--lib-path', help='library path', metavar='FILE', default='')
    parser.add_argument('--rsc-dir', help='resource directory', metavar='DIR', default='')
    parser.add_argument('--opt-str', help='option string (JSON format)', metavar='JSON', default='')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    parser.add_argument('--errpatch', help='analyze_bfr_errpatch', action='store_true')
    parser.add_argument('--set-log', help='set_log_levels')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
