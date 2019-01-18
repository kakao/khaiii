#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
TRIE 모듈
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import struct
import sys
from typing import List


#############
# constants #
#############
_NODE_STRUCT = struct.Struct('iIii')    # 필드: 문자, 값, 자식 노드가 시작되는 위치, 자식 노드의 갯수


#########
# types #
#########
class Node:
    """
    TRIE 노드
    """
    def __init__(self, char: str = ''):
        """
        Args:
            char:  (유니코드) 입력 문자
        """
        self.depth = -1    # 노드를 배열로 펼치기 위해 필요한 현재 노드의 루트로부터의 깊이
        self.char = char    # 입력 문자
        self.value = 0    # 단말 노드일 경우 그 값 (단말 노드가 아닌 경우 0)
        self.children = {}    # 자식 노드들
        self.child_start = -1    # 현재 노드로부터 자식 노드들의 시작점 사이에 있는 노드의 갯수

    def __str__(self):
        node_str = self.char if self.char else 'ROOT'
        if self.value > 0:
            node_str = '{%s: %d}' % (node_str, self.value)
        pfx = ''
        if self.depth > 0:
            pfx = '·' * self.depth
        child_start_str = ''
        if self.child_start > 0:
            child_start_str = ', {}'.format(self.child_start)
        return '{}({}, [{}]{})'.format(pfx, node_str, ', '.join(self.children.keys()),
                                       child_start_str)

    def insert(self, key: str, val: int):
        """
        문자열 키와 그 값을 현재 노드로부터 내려가며 적절한 위치에 삽입한다.
        Args:
            key:  키 문자열
            val:  값 (양수)
        """
        if val <= 0:
            raise ValueError('value must be greater than zero')
        if not key:
            self.value = val
        elif key[0] in self.children:
            self.children[key[0]].insert(key[1:], val)
        else:
            new_node = Node(key[0])
            self.children[key[0]] = new_node
            new_node.insert(key[1:], val)
        logging.debug('INSERT {%s: %d} INTO %s', key, val, self)

    def pack(self) -> bytes:
        """
        구조체로 packing한다.
        Returns:
            packing된 구조체
        """
        char = 0 if not self.char else self.char
        if isinstance(char, str):
            char = ord(char)
        return _NODE_STRUCT.pack(char, self.value, self.child_start, len(self.children))


class Trie:
    """
    TRIE 인터페이스
    """
    def __init__(self):
        self.root = Node()

    def insert(self, key: str, val: int):
        """
        하나의 문자열 키와 그 값을 삽입한다.
        Args:
            key:  키 문자열
            val:  값 (양수)
        """
        self.root.insert(key, val)

    def update(self, keys: List[str], vals: List[int] = None):
        """
        문자열 키와 그 값의 리스트를 차례로 삽입한다.
        값이 없을 경우 키 목록의 인덱스(1부터 시작)를 값으로 설정한다.
        Args:
            keys:  키 문자열 리스트
            vals:  값 (양수) 리스트
        """
        if vals:
            assert len(keys) == len(vals)
        else:
            vals = range(1, len(keys)+1)
        for key, val in zip(keys, vals):
            self.insert(key, val)

    def save(self, file_path: str):
        """
        TRIE 자료구조를 파일로 저장한다.
        Args:
            file_path:  파일 경로
        """
        with open(file_path, 'wb') as fout:
            nodes = self._breadth_first_traverse()
            for idx, node in enumerate(nodes):
                logging.debug('%d:%s', idx, node)
                fout.write(node.pack())
        logging.info('trie saved: %s', file_path)
        logging.info('total nodes: %d', len(nodes))
        logging.info('expected size: %d', len(nodes) * _NODE_STRUCT.size)

    def find(self, key: str) -> int:
        """
        키를 이용하여 값을 얻는다.
        Args:
            key:  키 문자열
        Returns:
            값(value, 양수). 키가 존재하지 않을 경우 0
        """
        node = self.root
        for char in key:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.value

    def _breadth_first_traverse(self) -> List[Node]:
        """
        너비우선으로 루트로부터 전체 노드를 탐색한다.
        Returns:
            노드의 리스트
        """
        self.root.depth = 0
        idx = 0
        nodes = [self.root, ]
        while idx < len(nodes):
            if (idx+1) % 1000000 == 0:
                logging.info('%dm-th node..', ((idx+1) / 1000000))
            node = nodes[idx]
            logging.debug('TRAVERSE: %s', node)
            for key in sorted(node.children.keys()):
                child_node = node.children[key]
                child_node.depth = node.depth + 1
                nodes.append(child_node)
            idx += 1
        self._set_child_start(nodes)
        return nodes

    @classmethod
    def _set_child_start(cls, nodes: List[Node]):
        """
        child_start 필드를 세팅한다.
        Args:
            nodes:  노드 리스트
        """
        for idx, node in enumerate(nodes):
            if idx == 0 or nodes[idx-1].depth != node.depth:
                partial_sum_of_children = 0
                num_of_next_siblings = 0
                for jdx in range(idx, len(nodes)):
                    if nodes[jdx].depth == node.depth:
                        num_of_next_siblings += 1
                    else:
                        break
            else:
                partial_sum_of_children += len(nodes[idx-1].children)
                num_of_next_siblings -= 1
            node.child_start = (partial_sum_of_children + num_of_next_siblings) if node.children \
                                                                                else -1


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    trie = Trie()
    for line_num, line in enumerate(sys.stdin, start=1):
        if line_num % 100000 == 0:
            logging.info('%dk-th line..', line_num // 1000)
        line = line.rstrip('\r\n')
        if not line:
            continue
        if args.has_val:
            try:
                key, val_str = line.rsplit('\t', 1)
                val = int(val_str)
            except ValueError:
                logging.error('%d-th line has no value: %s', line_num, line)
                continue
        else:
            key, val = line, line_num
        if len(key) > 900:
            logging.error('%d-th line is skipped because the key is too long: %s', line_num, key)
            continue
        trie.insert(key, val)
    trie.save(args.output)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='트라이를 빌드합니다.')
    parser.add_argument('-o', '--output', help='output file', metavar='FILE', required=True)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--val', dest='has_val', help='탭으로 구분된 마지막 컬럼이 값일 경우',
                        action='store_true')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
