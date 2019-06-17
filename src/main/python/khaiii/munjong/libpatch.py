# -*- coding: utf-8 -*-


"""
patch library for Sejong corpus
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from collections import namedtuple
import logging
import os
from typing import Dict, List, Tuple

from khaiii.munjong.sejong_corpus import SENT_OPEN_TAGS, SENT_CLOSE_TAGS, WORD_ID_PTN


#############
# constatns #
#############
# line types
WORD_TYPE = 1    # word (eojeol)
BOS_TYPE = 2    # begin of sentence markup
EOS_TYPE = 3    # end of sentence markup
MISC_TYPE = 0    # other miscellaneous lines


#########
# types #
#########
Line = namedtuple('Line', ['type_', 'wid', 'content'])


class Patch:    # pylint: disable=too-few-public-methods
    """
    patch line
    """
    # patch types (actions, operations)
    SENT_SPLIT = 1
    SENT_MERGE = 2
    WORD_REPLACE = 3
    WORD_INSERT = 4
    WORD_DELETE = 5
    # patch categories
    SENT_CATE = 1
    WORD_CATE = 2

    def __init__(self, type_: int, wid: str, content: str):
        """
        Args:
            type_:  patch type
            wid:  word ID
            content:  word content (tab separated second column)
        """
        self.type_ = type_
        self.wid = wid
        self.content = content

    def __eq__(self, other: 'Patch') -> bool:
        return self.type_ == other.type_ and self.wid == other.wid and self.content == other.content

    def __cmp__(self, other: 'Patch') -> int:
        if self.cate() == other.cate():
            if self.wid < other.wid:
                return -1
            if self.wid > other.wid:
                return 1
            return 0
        return self.cate() - other.cate()

    def __lt__(self, other: 'Patch') -> bool:
        return self.__cmp__(other) < 0

    def __str__(self) -> str:
        if self.type_ == self.WORD_REPLACE:
            return '=\t%s\t%s' % (self.wid, self.content)
        if self.type_ == self.WORD_INSERT:
            return '+\t%s\t%s' % (self.wid, self.content)
        if self.type_ == self.WORD_DELETE:
            return '-\t%s' % self.wid
        if self.type_ == self.SENT_SPLIT:
            return 'S\t%s\t%s' % (self.wid, self.content)
        if self.type_ == self.SENT_MERGE:
            return 'M\t%s\t%s' % (self.wid, self.content)
        raise RuntimeError('unknown patch type: %d' % self.type_)

    def cate(self) -> int:
        """
        get patch category. EOS/BOS patches are 0. word patches are 1
        Returns:
            category number
        """
        return self.SENT_CATE if self.type_ in [self.SENT_SPLIT, self.SENT_MERGE] \
                              else self.WORD_CATE

    @classmethod
    def parse(cls, line: str) -> 'Patch':
        """
        parse patch line
        Args:
            line:  patch line
        Returns:
            Patch object
        """
        cols = line.split('\t', 2)
        if len(cols) == 3:
            if cols[0] == '=':
                return Patch(cls.WORD_REPLACE, cols[1], cols[2])
            if cols[0] == '+':
                return Patch(cls.WORD_INSERT, cols[1], cols[2])
            if cols[0].upper() == 'M':
                return Patch(cls.SENT_MERGE, cols[1], cols[2])
            if cols[0].upper() == 'S':
                return Patch(cls.SENT_SPLIT, cols[1], cols[2])
        elif len(cols) == 2:
            if cols[0] == '-':
                return Patch(cls.WORD_DELETE, cols[1], '')
        raise RuntimeError('invalid patch format: %s' % line)


#############
# functions #
#############
def _load_corpus(path: str, enc: str) -> Tuple[List[Line], Dict[str, int]]:
    """
    load corpus
    Args:
        path:  file path
        enc:  file encoding
    Returns:
        list of lines
        word ID dic
    """
    lines = []
    wid_dic = {}
    for line in open(path, 'r', encoding=enc):
        line = line.rstrip('\r\n')
        if WORD_ID_PTN.match(line):
            wid, content = line.split('\t', 1)
            if wid in wid_dic:
                raise RuntimeError('duplicated word ID: %s' % line)
            wid_dic[wid] = len(lines)
            lines.append(Line(WORD_TYPE, wid, content))
        elif line in SENT_OPEN_TAGS:
            lines.append(Line(BOS_TYPE, None, line))
        elif line in SENT_CLOSE_TAGS:
            lines.append(Line(EOS_TYPE, None, line))
        else:
            lines.append(Line(MISC_TYPE, None, line))
            if line == '</tei.2>' and 'BTJO0443.txt' in path:
                break
    return lines, wid_dic


def _make_sent_patch(org_lines: List[Line], mod_lines: List[Line]) -> List[Patch]:
    """
    make EOS/BOS patch
    Args:
        org_lines:  original lines
        mod_lines:  modified lines
    Returns:
        EOS/BOS patches
    """
    def _get_eos_bos(lines: List[Line]) -> Tuple[str, str]:
        """
        get all EOS/BOS from lines
        Args:
            lines:  lines of corpus
        Returns:
            prev. word ID
            next word ID
        """
        eos_bos = []
        for idx, line in enumerate(lines):
            if line.type_ == EOS_TYPE and (idx+2) < len(lines) and lines[idx+1].type_ == BOS_TYPE \
                    and lines[idx-1].type_ == WORD_TYPE and lines[idx+2].type_ == WORD_TYPE:
                prev_wid = lines[idx-1].wid
                next_wid = lines[idx+2].wid
                eos_bos.append((prev_wid[:17], next_wid[:17]))
        return eos_bos

    org_eos_bos = set(_get_eos_bos(org_lines))
    mod_eos_bos = set(_get_eos_bos(mod_lines))
    patches = []
    for prev_wid, next_wid in org_eos_bos - mod_eos_bos:
        patches.append(Patch(Patch.SENT_MERGE, prev_wid, next_wid))
    for prev_wid, next_wid in mod_eos_bos - org_eos_bos:
        patches.append(Patch(Patch.SENT_SPLIT, prev_wid, next_wid))
    return sorted(patches)


def _make_word_patch(org_lines: List[Line], org_wid_dic: Dict[str, int], mod_lines: List[Line],
                     mod_wid_dic: Dict[str, int]) -> List[Patch]:
    """
    make word patch
    Args:
        org_lines:  original lines
        org_wid_dic:  original word ID dictionary
        mod_lines:  modified lines
        mod_wid_dic:  modified word ID dictionary
    Returns:
        word patches
    """
    patches = []
    for org_line in org_lines:
        if org_line.type_ != WORD_TYPE:
            continue
        if org_line.wid in mod_wid_dic:
            mod_line = mod_lines[mod_wid_dic[org_line.wid]]
            if org_line.content != mod_line.content:
                patches.append(Patch(Patch.WORD_REPLACE, mod_line.wid, mod_line.content))
        else:
            patches.append(Patch(Patch.WORD_DELETE, org_line.wid, ''))
    for mod_line in mod_lines:
        if mod_line.type_ != WORD_TYPE or mod_line.wid in org_wid_dic:
            continue
        patches.append(Patch(Patch.WORD_INSERT, mod_line.wid, mod_line.content))
    return sorted(patches)


def make(org_path: str, org_enc: str, mod_path: str, mod_enc: str) -> List[Patch]:
    """
    make patch from two file
    Args:
        org_path:  original file path
        org_enc:  original file encoding
        mod_path:  modified file path
        mod_enc:  modified file encoding
    Returns:
        patch contents (list of patch lines)
    """
    org_lines, org_wid_dic = _load_corpus(org_path, org_enc)
    mod_lines, mod_wid_dic = _load_corpus(mod_path, mod_enc)

    sent_patches = _make_sent_patch(org_lines, mod_lines)
    word_patches = _make_word_patch(org_lines, org_wid_dic, mod_lines, mod_wid_dic)
    return sent_patches + word_patches


def _load_patch(patch_path: str) -> List[Patch]:
    """
    load patch from file
    Args:
        patch_path:  patch path
    Returns:
        patches
    """
    patch_name = os.path.basename(patch_path)
    wid_dic = set()
    patches = []
    for line_num, line in enumerate(open(patch_path, 'r', encoding='UTF-8'), start=1):
        line = line.rstrip('\r\n')
        if not line:
            continue
        patch = Patch.parse(line)
        if patch.cate() == Patch.SENT_CATE:
            if (patch.wid, patch.content) in wid_dic:
                logging.error('%s(%d): patch conflict: %s', patch_name, line_num, line)
            else:
                patches.append(patch)
                wid_dic.add((patch.wid, patch.content))
        else:
            if patch.wid in wid_dic:
                logging.error('%s(%d): patch conflict: %s', patch_name, line_num, line)
            else:
                patches.append(patch)
                wid_dic.add(patch.wid)
    return patches


def _apply_sent_merge_patch(org_lines: List[Line], patches: List[Patch]) -> List[Line]:
    """
    apply merge EOS/BOS patches
    Args:
        org_lines:  original lines
        patches:  patches
    Returns:
        modified lines
    """
    merge_patches = {patch.wid: patch.content for patch in patches \
                     if patch.type_ == Patch.SENT_MERGE}
    mod_lines = []
    idx = 0
    while idx < len(org_lines):
        org_line = org_lines[idx]
        if org_line.type_ == EOS_TYPE and org_lines[idx+1].type_ == BOS_TYPE and \
                        org_lines[idx-1].type_ == WORD_TYPE and org_lines[idx+2].type_ == WORD_TYPE:
            prev_wid = org_lines[idx-1].wid[:17]
            next_wid = org_lines[idx+2].wid[:17]
            if prev_wid in merge_patches and merge_patches[prev_wid] == next_wid:
                del merge_patches[prev_wid]
                idx += 2
                continue
        mod_lines.append(org_line)
        idx += 1
    if merge_patches:
        for prev_wid, next_wid in merge_patches.items():
            logging.error('remaining merge sentence patches: %s\t%s', prev_wid, next_wid)
    return mod_lines


def _build_wid_dic(lines: List[Line]) -> Dict[str, int]:
    """
    build word ID dictionary
    Args:
        lines:  lines of corpus
    Returns:
        word ID dictionary
    """
    return {line.wid: idx for idx, line in enumerate(lines) if line.type_ == WORD_TYPE}


def _apply_sent_split_patch(mod_lines: List[Line], patches: List[Patch]) -> List[Line]:
    """
    apply split EOS/BOS patches
    Args:
        mod_lines:  modified lines
        patches:  patches
    Returns:
        modified lines
    """
    mod_wid_dic = _build_wid_dic(mod_lines)
    split_patches = {patch.wid: patch.content for patch in patches \
                     if patch.type_ == Patch.SENT_SPLIT}
    for prev_wid, next_wid in sorted(split_patches.items(), key=lambda x: x[0], reverse=True):
        idx = mod_wid_dic[next_wid]
        assert mod_lines[idx].wid == next_wid
        if mod_lines[idx-1].type_ == WORD_TYPE and mod_lines[idx-1].wid[:17] == prev_wid:
            mod_lines.insert(idx, Line(BOS_TYPE, None, '<p>'))
            mod_lines.insert(idx, Line(EOS_TYPE, None, '</p>'))
            del split_patches[prev_wid]
    if split_patches:
        for prev_wid, next_wid in split_patches.items():
            logging.error('remaining split sentence patches: %s\t%s', prev_wid, next_wid)
    return mod_lines


def _apply_word_del_rep_patch(mod_lines: List[Line], patches: List[Patch]) -> List[Line]:
    """
    apply word delete/replace patches
    Args:
        mod_lines:  modified lines
        patches:  patches
    Returns:
        modified lines
    """
    delete_patches = {patch.wid for patch in patches if patch.type_ == Patch.WORD_DELETE}
    replace_patches = {patch.wid: patch.content for patch in patches \
                       if patch.type_ == Patch.WORD_REPLACE}
    new_lines = []
    for line in mod_lines:
        if line.type_ == WORD_TYPE:
            if line.wid in delete_patches:
                delete_patches.remove(line.wid)
                continue
            elif line.wid in replace_patches:
                new_lines.append(Line(WORD_TYPE, line.wid, replace_patches[line.wid]))
                del replace_patches[line.wid]
                continue
        new_lines.append(line)
    if delete_patches:
        for wid in delete_patches:
            logging.error('remaining delete word patches: %s', wid)
    if replace_patches:
        for wid, content in replace_patches.items():
            logging.error('remaining replace word patches: %s\t%s', wid, content)
    return new_lines


def _apply_word_insert_patch(mod_lines: List[Line], patches: List[Patch]) -> List[Line]:
    """
    apply word insert patches
    Args:
        mod_lines:  modified lines
        patches:  patches
    Returns:
        modified lines
    """
    insert_patches = sorted([(patch.wid, patch.content) for patch in patches \
                             if patch.type_ == Patch.WORD_INSERT])
    prev_idx = -1
    curr_idx = 0
    while curr_idx < len(mod_lines) and insert_patches:
        curr_word = mod_lines[curr_idx]
        if curr_word.type_ != WORD_TYPE:
            curr_idx += 1
            continue
        wid, content = insert_patches[0]
        if curr_word.wid < wid:
            prev_idx = curr_idx
            curr_idx += 1
            continue
        mod_lines.insert(prev_idx+1, Line(WORD_TYPE, wid, content))
        del insert_patches[0]
        prev_idx += 1
        curr_idx += 1
    return mod_lines


def apply(org_path: str, org_enc: str, patch_path: str, mod_path: str, mod_enc: str):
    """
    apply path to original corpus then get modified corpus
    Args:
        org_path:  original file path
        org_enc:  original file encoding
        patch_path:  patch file path
        mod_path:  modified file path
        mod_enc:  modified file encoding
    """
    patches = _load_patch(patch_path)
    org_lines, _ = _load_corpus(org_path, org_enc)
    mod_lines = _apply_sent_merge_patch(org_lines, patches)
    mod_lines = _apply_word_del_rep_patch(mod_lines, patches)
    mod_lines = _apply_sent_split_patch(mod_lines, patches)
    mod_lines = _apply_word_insert_patch(mod_lines, patches)
    with open(mod_path, 'w', encoding=mod_enc) as fout:
        for mod_line in mod_lines:
            if mod_line.type_ == WORD_TYPE:
                print('%s\t%s' % (mod_line.wid, mod_line.content), file=fout)
            else:
                print(mod_line.content, file=fout)
