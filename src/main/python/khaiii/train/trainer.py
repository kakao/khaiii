# -*- coding: utf-8 -*-


"""
training related library
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2020-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
from datetime import datetime, timedelta
import json
import logging
import os
import pathlib
import pprint
import sys
from typing import List, Tuple

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Dataset
from tqdm import tqdm

from khaiii.train import dataset
from khaiii.train.dataset import BatchIter, FIELDS, SentIter
from khaiii.train.evaluator import Evaluator
from khaiii.train.models import Model
from khaiii.resource.resource import Resource


#############
# constants #
#############
_LOG = logging.getLogger(__name__)

# if it is true, overfittig mode is enabled for debugging
# train, valid, test are all same tiny dataset
# evaluation metric is set to loss to minimize
_OVERFITTING_MODE = False


#############
# functions #
#############
class Trainer:
    """
    trainer class
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            cfg:  config
        """
        self.cfg = cfg
        setattr(cfg, 'model_id', self.model_id(cfg))
        setattr(cfg, 'out_dir', '{}/{}'.format(cfg.logdir, cfg.model_id))
        setattr(cfg, 'context_len', 2 * cfg.window + 1)
        self.rsc = Resource(cfg)
        self._load_dataset()
        setattr(cfg, 'vocab_in', FIELDS['char'].vocab_size())
        setattr(cfg, 'vocab_out', FIELDS['tag'].vocab_size())
        self.model = Model(cfg)
        if torch.cuda.is_available() and cfg.gpu_num >= 0:
            self.model.cuda(device=cfg.gpu_num)
            self.device = torch.device(f'cuda:{cfg.gpu_num}')    # pylint: disable=no-member
        else:
            self.device = torch.device('cpu')    # pylint: disable=no-member
        self.optimizer = torch.optim.Adam(self.model.parameters(), cfg.learning_rate)
        self.criterion_pos = nn.CrossEntropyLoss(ignore_index=FIELDS['tag'].vocab.stoi['<pad>'])
        self.criterion_spc = nn.CrossEntropyLoss()
        self.evaler = Evaluator()
        if 'epoch' not in cfg.__dict__:
            setattr(cfg, 'epoch', 0)
            setattr(cfg, 'best_epoch', 0)
        self.log_file = None    # tab separated log file
        self.sum_wrt = None    # tensorboard summary writer
        self.loss_trains = []
        self.loss_valids = []
        self.acc_chars = []
        self.acc_words = []
        self.f_scores = []
        self.learning_rates = []
        self.spc_idx = torch.tensor([self.cfg.window, ], device=self.device)    # pylint: disable=not-callable

    @classmethod
    def model_id(cls, cfg: Namespace) -> str:
        """
        get model ID
        Args:
            cfg:  config
        Returns:
            model ID
        """
        if cfg.input:
            data_name = os.path.basename(cfg.input).split('.')[0]
        else:
            data_name = 'STDIN'
        model_cfgs = [
            data_name,
            'minfrq{}'.format(cfg.min_freq),
            'win{}'.format(cfg.window),
            'hdo{}'.format(cfg.hdn_dropout),
            'sdo{}'.format(cfg.spc_dropout),
            'emb{}'.format(cfg.embed_dim),
            'lr{}'.format(cfg.learning_rate),
            'lrd{}'.format(cfg.lr_decay),
            'bs{}'.format(cfg.batch_size),
        ]
        return '.'.join(model_cfgs)

    def _load_dataset(self):
        """
        load training dataset
        """
        data = dataset.load(sys.stdin, self.cfg.window, self.rsc)
        if _LOG.isEnabledFor(logging.DEBUG):
            for sent in data.examples:
                _LOG.debug(sent)
        if _OVERFITTING_MODE:
            self.data_train = self.data_test = self.data_valid = data
        else:
            self.data_train, self.data_test, self.data_valid = \
                data.split(split_ratio=[0.9, 0.05, 0.05])
        for sent in self.data_valid.examples:
            sent.set_tags(restore_dic=self.rsc.restore_dic)

    @classmethod
    def _dt_str(cls, dt_obj: datetime) -> str:
        """
        string formatting for datetime object
        Args:
            dt_obj:  datetime object
        Returns:
            string
        """
        return dt_obj.strftime('%m/%d %H:%M:%S')

    @classmethod
    def _elapsed(cls, td_obj: timedelta) -> str:
        """
        string formatting for timedelta object
        Args:
            td_obj:  timedelta object
        Returns:
            string
        """
        seconds = td_obj.seconds
        if td_obj.days > 0:
            seconds += td_obj.days * 24 * 3600
        hours = seconds // 3600
        seconds -= hours * 3600
        minutes = seconds // 60
        seconds -= minutes * 60
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)

    def _restore_prev_train(self):
        """
        restore previous state to continue training if stopped training
        """
        out_path = pathlib.Path(self.cfg.out_dir)
        cfg_path = pathlib.Path('{}/config.json'.format(self.cfg.out_dir))
        if not out_path.is_dir() or not cfg_path.is_file():
            return
        _LOG.info('==== continue training: %s ====', self.cfg.model_id)
        cfg = json.load(open(cfg_path, 'r', encoding='UTF-8'))
        gpu_num = self.cfg.gpu_num
        for key, val in cfg.items():
            setattr(self.cfg, key, val)
        setattr(self.cfg, 'gpu_num', gpu_num)
        self._revert_to_best(False)

        f_score_best = 0.0
        best_idx = -1
        for idx, line in enumerate(open('{}/log.tsv'.format(self.cfg.out_dir))):
            line = line.rstrip('\r\n')
            if not line:
                continue
            (epoch, loss_train, loss_valid, acc_char, acc_word, f_score, learning_rate) \
                = line.split('\t')
            self.cfg.epoch = int(epoch) + 1
            self.cfg.best_epoch = self.cfg.epoch
            self.loss_trains.append(float(loss_train))
            self.loss_valids.append(float(loss_valid))
            self.acc_chars.append(float(acc_char))
            self.acc_words.append(float(acc_word))
            self.f_scores.append(float(f_score))
            self.learning_rates.append(float(learning_rate))
            if float(f_score) > f_score_best:
                f_score_best = float(f_score)
                best_idx = idx
        _LOG.info('---- [%d] los(trn/val): %.4f / %.4f, acc(chr/wrd): %.4f / %.4f, ' \
                  'f-score: %.4f, lr: %.8f ----', self.cfg.epoch, self.loss_trains[best_idx],
                  self.loss_valids[best_idx], self.acc_chars[best_idx], self.acc_words[best_idx],
                  self.f_scores[best_idx], self.learning_rates[-1])

    def train(self):
        """
        train model with dataset
        """
        self._restore_prev_train()
        _LOG.info('config: %s', pprint.pformat(self.cfg.__dict__))

        train_begin = datetime.now()
        _LOG.info('{{{{ training begin: %s {{{{', self._dt_str(train_begin))
        pathlib.Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open('{}/log.tsv'.format(self.cfg.out_dir), 'at')
        self.sum_wrt = SummaryWriter(self.cfg.out_dir)
        patience = self.cfg.patience
        for _ in range(100000):
            is_best = self._train_epoch()
            if is_best:
                patience = self.cfg.patience
                continue
            if patience <= 0:
                break
            self._revert_to_best(True)
            patience -= 1
            _LOG.info('==== revert to EPOCH[%d], f-score: %.4f, patience: %d ====',
                      self.cfg.best_epoch, max(self.f_scores), patience)

        train_end = datetime.now()
        train_elapsed = self._elapsed(train_end - train_begin)
        _LOG.info('}}}} training end: %s, elapsed: %s, epoch: %s }}}}',
                  self._dt_str(train_end), train_elapsed, self.cfg.epoch)

        for sent in self.data_test.examples:
            sent.set_tags(restore_dic=self.rsc.restore_dic)
        avg_loss, acc_char, acc_word, f_score = self.evaluate(self.data_test)
        _LOG.info('==== test loss: %.4f, char acc: %.4f, word acc: %.4f, f-score: %.4f ====',
                  avg_loss, acc_char, acc_word, f_score)

    def _revert_to_best(self, is_decay_lr: bool):
        """
        revert to previous best model
        Args:
            is_decay_lr:  whether multiply decay factor or not
        """
        self.model.load('{}/model.state'.format(self.cfg.out_dir))
        if is_decay_lr:
            self.cfg.learning_rate *= self.cfg.lr_decay
        self._load_optim(self.cfg.learning_rate)

    def _train_epoch(self) -> bool:
        """
        train a single epoch
        Returns:
            whether this epoch got best or not
        """
        self.model.train()
        loss_trains = []
        train_iter = BatchIter(self.data_train, self.cfg.batch_size, device=self.device)
        progress = tqdm(train_iter, 'EPOCH[{}]'.format(self.cfg.epoch), mininterval=1, ncols=100)
        update_interval = max(len(train_iter) // 100, 10)
        for step, train_batch in enumerate(progress, start=1):
            if step % update_interval == 0:
                avg_loss = sum(loss_trains) / len(loss_trains)
                progress.set_description('EPOCH[{}] ({:.6f})'.format(self.cfg.epoch, avg_loss))

            if _LOG.isEnabledFor(logging.DEBUG):
                for idx, sent in enumerate(train_batch.char):
                    char_str = FIELDS['char'].itos(sent)
                    _LOG.debug('\tsent[%d]: %s', idx, char_str)
                _LOG.debug('char:\n%s', train_batch.char)
                for idx, sent in enumerate(train_batch.tag):
                    tag_str = FIELDS['tag'].itos(sent)
                    _LOG.debug('\tsent[%d]: %s', idx, tag_str)
                _LOG.debug('tag:\n%s', train_batch.tag)
                _LOG.debug('left_spc:\n%s', train_batch.left_spc)
                _LOG.debug('right_spc:\n%s', train_batch.right_spc)

            batch_logits_pos, batch_logits_spc = self.model(train_batch)
            loss_train_pos = self.criterion_pos(
                batch_logits_pos.view(-1, batch_logits_pos.size()[-1]),
                train_batch.tag.view(-1))
            right_spc = torch.index_select(train_batch.right_spc, 2, self.spc_idx)    # pylint: disable=no-member
            loss_train_spc = self.criterion_spc(
                batch_logits_spc.view(-1, batch_logits_spc.size()[-1]),
                right_spc.view(-1))
            loss_train = loss_train_pos + loss_train_spc
            loss_trains.append(loss_train.item())
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()

        avg_loss_valid, acc_char, acc_word, f_score = self.evaluate(self.data_valid)
        is_best = self._check_epoch(loss_trains, avg_loss_valid, acc_char, acc_word, f_score)
        self.cfg.epoch += 1
        return is_best

    def _check_epoch(self, loss_trains: List[float], avg_loss_valid: float,
                     acc_char: float, acc_word: float, f_score: float) -> bool:
        """
        check step for every eopch
        Args:
            loss_trains:  list of loses in train dataset
            avg_loss_valid:  average loss on valid dataset
            acc_char:  character accuracy
            acc_word:  word accuracy
            f_score:  f-score
        Returns:
            whether this epoch got best or not
        """
        avg_loss_train = sum(loss_trains) / len(loss_trains)
        loss_trains.clear()
        self.loss_trains.append(avg_loss_train)
        self.loss_valids.append(avg_loss_valid)
        self.acc_chars.append(acc_char)
        self.acc_words.append(acc_word)
        self.f_scores.append(f_score)
        self.learning_rates.append(self.cfg.learning_rate)
        is_best = self._is_best()
        is_best_str = 'BEST' if is_best else '< {:.4f}'.format(max(self.f_scores))
        _LOG.info('[Los trn]  [Los vld]  [Acc chr]  [Acc wrd]  [F-score]' \
                  '           [LR]')
        _LOG.info('{:9.4f}  {:9.4f}  {:9.4f}  {:9.4f}  {:9.4f} {:8}  {:.8f}' \
                .format(avg_loss_train, avg_loss_valid, acc_char, acc_word, f_score, is_best_str,
                        self.cfg.learning_rate))
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.cfg.epoch, avg_loss_train, avg_loss_valid,
                                                  acc_char, acc_word, f_score,
                                                  self.cfg.learning_rate),
              file=self.log_file)
        self.log_file.flush()
        self.sum_wrt.add_scalar('loss-train', avg_loss_train, self.cfg.epoch)
        self.sum_wrt.add_scalar('loss-val', avg_loss_valid, self.cfg.epoch)
        self.sum_wrt.add_scalar('acc-char', acc_char, self.cfg.epoch)
        self.sum_wrt.add_scalar('acc-word', acc_word, self.cfg.epoch)
        self.sum_wrt.add_scalar('f-score', f_score, self.cfg.epoch)
        self.sum_wrt.add_scalar('learning-rate', self.cfg.learning_rate, self.cfg.epoch)
        return is_best

    def _is_best(self) -> bool:
        """
        whether this epoch got best or not. if so, save current state
        Returns:
            whether the last score is the best or not
        """
        if _OVERFITTING_MODE:
            if len(self.loss_valids) > 1 and min(self.loss_valids[:-1]) <= self.loss_valids[-1]:
                return False
        else:
            if len(self.f_scores) > 1 and max(self.f_scores[:-1]) >= self.f_scores[-1]:
                return False
        # this epoch hits new max value
        self.cfg.best_epoch = self.cfg.epoch
        self.model.save('{}/model.state'.format(self.cfg.out_dir))
        torch.save(self.optimizer.state_dict(), '{}/optimizer.state'.format(self.cfg.out_dir))
        with open('{}/config.json'.format(self.cfg.out_dir), 'w', encoding='UTF-8') as fout:
            json.dump(vars(self.cfg), fout, indent=2, sort_keys=True)
        return True

    def _load_optim(self, learning_rate: float):
        """
        load optimizer parameters
        Args:
            learning_rate:  learning rate
        """
        path = '{}/optimizer.state'.format(self.cfg.out_dir)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.optimizer.load_state_dict(state_dict)
        self.optimizer.param_groups[0]['lr'] = learning_rate

    def evaluate(self, data: Dataset) -> Tuple[float, float, float, float]:
        """
        evaluate f-score
        Args:
            data:  gold standard data
        Returns:
            average valid loss
            character accuracy
            word accuracy
            f-score
        """
        self.model.eval()
        loss_evals = []
        eval_iter = SentIter(data, device=self.device)
        progress = tqdm(eval_iter, ' EVAL[{}]'.format(self.cfg.epoch), mininterval=1, ncols=100)
        update_interval = max(len(eval_iter) // 10, 10)
        tag_vocab = FIELDS['tag'].vocab.itos
        for step, (eval_batch, sent) in enumerate(progress, start=1):
            if step % update_interval == 0:
                avg_loss = sum(loss_evals) / len(loss_evals)
                progress.set_description(' EVAL[{}] ({:.6f})'.format(self.cfg.epoch, avg_loss))
            with torch.no_grad():
                batch_logits_pos, batch_logits_spc = self.model(eval_batch)
                loss_train_pos = self.criterion_pos(
                    batch_logits_pos.view(-1, batch_logits_pos.size()[-1]),
                    eval_batch.tag.view(-1))
                right_spc = torch.index_select(eval_batch.right_spc, 2, self.spc_idx)    # pylint: disable=no-member
                loss_train_spc = self.criterion_spc(
                    batch_logits_spc.view(-1, batch_logits_spc.size()[-1]),
                    right_spc.view(-1))
                loss_eval = loss_train_pos + loss_train_spc
                loss_evals.append(loss_eval.item())

                _, predicts = F.softmax(batch_logits_pos[0], dim=1).max(1)
                pred_tags = [tag_vocab[t.item()] for t in predicts]
                pred_sent = sent.copy()
                pred_sent.set_tags(pred_tags, self.rsc.restore_dic)
                self.evaler.count(sent, pred_sent)
        avg_loss = sum(loss_evals) / len(loss_evals)
        char_acc, word_acc, f_score = self.evaler.evaluate()
        return avg_loss, char_acc, word_acc, f_score
