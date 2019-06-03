# -*- coding: utf-8 -*-


"""
evaluation related module
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from collections import Counter
import logging
from typing import List, TextIO, Tuple

from khaiii.train.sentence import PosMorph, PosSentence, PosWord


#########
# types #
#########
class Evaluator:
    """
    evauator
    """
    def __init__(self):
        self.cnt = Counter()

    def evaluate(self) -> Tuple[float, float, float]:
        """
        char/word accuracy, f-score(recall/precision)를 측정한다.
        Returns:
            character accuracy
            word accuracy
            f-score
        """
        char_acc = self.cnt['match_chars'] / self.cnt['total_chars']
        word_acc = self.cnt['match_words'] / self.cnt['total_words']
        if self.cnt['match_morphs'] == 0:
            recall = precision = f_score = 0.0
        else:
            recall = self.cnt['match_morphs'] / self.cnt['total_gold_morphs']
            precision = self.cnt['match_morphs'] / self.cnt['total_pred_morphs']
            f_score = 2.0 * recall * precision / (recall + precision)
        self.cnt.clear()
        return char_acc, word_acc, f_score

    def count(self, correct_sent: PosSentence, predict_sent: PosSentence):
        """
        정답 문장과 비교하여 맞춘 갯수를 샌다.
        Args:
            correct_sent:  정답 문장
            predict_sent:  예측한 문장
        """
        assert len(correct_sent.words) == len(predict_sent.words)
        for gold, pred in zip(correct_sent.pos_tagged_words, predict_sent.pos_tagged_words):
            self.cnt['total_chars'] += len(gold.res_tags)
            self.cnt['match_chars'] += len([1 for x, y in zip(gold.res_tags, pred.res_tags)
                                            if x == y])
            self._count_word(gold, pred)

    def _count_word(self, gold: PosWord, pred: PosWord):
        """
        count with gold standard and predicted (will update counter)
        Args:
            gold:  gold standard word
            pred:  predicted word
        """
        self.cnt['total_words'] += 1
        gold_morphs = gold.pos_tagged_morphs
        pred_morphs = pred.pos_tagged_morphs
        if gold == pred:
            self.cnt['match_words'] += 1
            num_match = len(gold_morphs)
            self.cnt['total_gold_morphs'] += num_match
            self.cnt['total_pred_morphs'] += num_match
            self.cnt['match_morphs'] += num_match
            return
        logging.debug('gold: %s', ' '.join([str(_) for _ in gold_morphs]))
        logging.debug('pred: %s', ' '.join([str(_) for _ in pred_morphs]))
        self.cnt['total_gold_morphs'] += len(gold_morphs)
        self.cnt['total_pred_morphs'] += len(pred_morphs)
        gold_set = self.morphs_to_set(gold_morphs)
        pred_set = self.morphs_to_set(pred_morphs)
        self.cnt['match_morphs'] += len(gold_set & pred_set)

    @classmethod
    def morphs_to_set(cls, morphs: List[PosMorph]) -> set:
        """
        make set from morpheme list
        Args:
            morphs:  morpheme list
        Returns:
            morphemes set
        """
        morph_cnt = Counter([(morph.morph, morph.pos_tag) for morph in morphs])
        morph_set = set()
        for (lex, tag), freq in morph_cnt.items():
            if freq == 1:
                morph_set.add((lex, tag))
            else:
                morph_set.update([(lex, tag, _) for _ in range(1, freq+1)])
        return morph_set

    def report(self, fout: TextIO):
        """
        report recall/precision to file
        Args:
            fout:  output file
        """
        print('word accuracy: %d / %d = %.4f' % (self.cnt['match_words'], self.cnt['total_words'],
                                                 self.cnt['match_words'] / self.cnt['total_words']),
              file=fout)
        if self.cnt['match_morphs'] == 0:
            recall = precision = f_score = 0.0
        else:
            recall = self.cnt['match_morphs'] / self.cnt['total_gold_morphs']
            precision = self.cnt['match_morphs'] / self.cnt['total_pred_morphs']
            f_score = 2.0 * recall * precision / (recall + precision)
        print('f-score / (recall, precision): %.4f / (%.4f, %.4f)' % (f_score, recall, precision),
              file=fout)
