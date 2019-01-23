# -*- coding: utf-8 -*-


"""
training related library
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import copy
from datetime import datetime, timedelta
import json
import logging
import os
import pathlib
import pprint
from typing import Iterator, List, Tuple

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from khaiii.train.dataset import PosDataset, PosSentTensor
from khaiii.train.evaluator import Evaluator
from khaiii.train.models import CnnModel
from khaiii.resource.resource import Resource


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
        self.model = CnnModel(cfg, self.rsc)
        self.optimizer = torch.optim.Adam(self.model.parameters(), cfg.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.evaler = Evaluator()
        self._load_dataset()
        if 'step' not in cfg.__dict__:
            setattr(cfg, 'step', 0)
            setattr(cfg, 'best_step', 0)
        self.log_file = None    # tab separated log file
        self.sum_wrt = None    # tensorboard summary writer
        self.loss_trains = []
        self.loss_devs = []
        self.acc_chars = []
        self.acc_words = []
        self.f_scores = []
        self.learning_rates = []

    @classmethod
    def model_id(cls, cfg: Namespace) -> str:
        """
        get model ID
        Args:
            cfg:  config
        Returns:
            model ID
        """
        model_cfgs = [
            os.path.basename(cfg.in_pfx),
            'cut{}'.format(cfg.cutoff),
            'win{}'.format(cfg.window),
            'sdo{}'.format(cfg.spc_dropout),
            'emb{}'.format(cfg.embed_dim),
            'lr{}'.format(cfg.learning_rate),
            'lrd{}'.format(cfg.lr_decay),
            'bs{}'.format(cfg.batch_size),
            'cs{}'.format(cfg.check_step),
        ]
        return '.'.join(model_cfgs)

    def _load_dataset(self):
        """
        load training dataset
        """
        dataset_dev_path = '{}.dev'.format(self.cfg.in_pfx)
        self.dataset_dev = PosDataset(self.cfg, self.rsc.restore_dic,
                                      open(dataset_dev_path, 'r', encoding='UTF-8'))
        dataset_train_path = '{}.train'.format(self.cfg.in_pfx)
        self.dataset_train = PosDataset(self.cfg, self.rsc.restore_dic,
                                        open(dataset_train_path, 'r', encoding='UTF-8'))

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
        기존에 학습하다 중지한 경우 그 이후부터 계속해서 학습할 수 있도록 이전 상태를 복원한다.
        """
        out_path = pathlib.Path(self.cfg.out_dir)
        cfg_path = pathlib.Path('{}/config.json'.format(self.cfg.out_dir))
        if not out_path.is_dir() or not cfg_path.is_file():
            return
        logging.info('==== continue training: %s ====', self.cfg.model_id)
        cfg = json.load(open(cfg_path, 'r', encoding='UTF-8'))
        for key, val in cfg.items():
            setattr(self.cfg, key, val)
        self._revert_to_best(False)

        f_score_best = 0.0
        best_idx = -1
        for idx, line in enumerate(open('{}/log.tsv'.format(self.cfg.out_dir))):
            line = line.rstrip('\r\n')
            if not line:
                continue
            (step, loss_train, loss_dev, acc_char, acc_word, f_score, learning_rate) \
                    = line.split('\t')
            self.cfg.step = self.cfg.best_step = int(step) * self.cfg.check_step
            self.loss_trains.append(float(loss_train))
            self.loss_devs.append(float(loss_dev))
            self.acc_chars.append(float(acc_char))
            self.acc_words.append(float(acc_word))
            self.f_scores.append(float(f_score))
            self.learning_rates.append(float(learning_rate))
            if float(f_score) > f_score_best:
                f_score_best = float(f_score)
                best_idx = idx
        logging.info('---- [%d] los(trn/dev): %.4f / %.4f, acc(chr/wrd): %.4f / %.4f, ' \
                     'f-score: %.4f, lr: %.8f ----', self.cfg.step // self.cfg.check_step,
                     self.loss_trains[best_idx], self.loss_devs[best_idx], self.acc_chars[best_idx],
                     self.acc_words[best_idx], self.f_scores[best_idx], self.learning_rates[-1])

    @classmethod
    def _inf_data_iterator(cls, dataset: PosDataset) -> Iterator[PosSentTensor]:
        """
        데이터셋을 무한히 반복하여 문장을 출력하는 제너레이터
        Args:
            dataset:  데이터셋
        Yields:
            PosSentTensor 객체
        """
        for _ in range(1000000):
            for sent in dataset:
                yield sent

    def train(self):
        """
        train model with dataset
        """
        self._restore_prev_train()
        logging.info('config: %s', pprint.pformat(self.cfg.__dict__))

        train_begin = datetime.now()
        logging.info('{{{{ training begin: %s {{{{', self._dt_str(train_begin))
        if torch.cuda.is_available():
            self.model.cuda()
        pathlib.Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open('{}/log.tsv'.format(self.cfg.out_dir), 'at')
        self.sum_wrt = SummaryWriter(self.cfg.out_dir)
        check_start = (1 if self.cfg.step == 0 else (self.cfg.step // self.cfg.check_step + 1))
        patience = self.cfg.patience
        train_iter = self._inf_data_iterator(self.dataset_train)
        for check_id in range(check_start, 1000000):
            is_best = self._train_and_check(check_id, train_iter)
            if is_best:
                patience = self.cfg.patience
                continue
            if patience <= 0:
                break
            self._revert_to_best(True)
            patience -= 1
            tqdm.write('==== revert to check: {}, f-score: {:.4f}, patience: {} ===='.format( \
                    self.cfg.best_step // self.cfg.check_step, max(self.f_scores), patience))

        train_end = datetime.now()
        train_elapsed = self._elapsed(train_end - train_begin)
        logging.info('}}}} training end: %s, elapsed: %s, step: %dk }}}}',
                     self._dt_str(train_end), train_elapsed, self.cfg.step // 1000)

    def _revert_to_best(self, is_decay_lr: bool):
        """
        이전 best 모델로 되돌린다.
        Args:
            is_decay_lr:  whether multiply decay factor or not
        """
        self.model.load('{}/model.state'.format(self.cfg.out_dir))
        if is_decay_lr:
            self.cfg.learning_rate *= self.cfg.lr_decay
        self._load_optim('{}/optim.state'.format(self.cfg.out_dir), self.cfg.learning_rate)

    def _train_and_check(self, check_id: int, train_iter: Iterator[PosSentTensor]) -> bool:
        """
        cfg.check_step 만큼의 step을 수행하고 evaluation을 수행한다.
        Args:
            check_id:  check ID
            train_iter:  학습 데이터 iterator
        Returns:
            best f-score를 기록한 step 여부
        """
        start_step = self.cfg.step
        loss_batch = torch.tensor(0.0)    # pylint: disable=not-callable
        batch_size = 0
        loss_trains = []
        train_sents = tqdm(train_iter, '[{}]'.format(check_id), mininterval=1, ncols=100)
        for train_sent in train_sents:
            train_labels, train_contexts = train_sent.to_tensor(self.cfg, self.rsc, True)
            if torch.cuda.is_available():
                train_labels = train_labels.cuda()
                train_contexts = train_contexts.cuda()
                loss_batch = loss_batch.cuda()

            self.model.train()
            train_outputs = self.model(train_contexts)
            train_outputs.requires_grad_()
            loss_train = self.criterion(train_outputs, train_labels)
            loss_train.backward()
            loss_trains.append(loss_train.item())
            loss_batch += loss_train
            batch_size += len(train_labels)
            if batch_size < self.cfg.batch_size:
                continue

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.sum_wrt.add_scalar('loss-batch', loss_batch.item(), self.cfg.step)
            self.cfg.step += 1
            loss_batch = torch.tensor(0.0)    # pylint: disable=not-callable
            batch_size = 0

            if (self.cfg.step - start_step) >= self.cfg.check_step:
                train_sents.close()
                break

        avg_loss_dev, acc_char, acc_word, f_score = self.evaluate()
        return self._check(check_id, loss_trains, avg_loss_dev, acc_char, acc_word, f_score)

    def _check(self, check_id: int, loss_trains: List[float], avg_loss_dev: float, acc_char: float,
               acc_word: float, f_score: float) -> bool:
        """
        cfg.check_step번의 step마다 수행하는 체크
        Args:
            check_id:  check ID
            loss_trains:  train 코퍼스에서 각 배치별 loss 리스트
            avg_loss_dev:  dev 코퍼스 문장 별 평균 loss
            acc_char:  음절 정확도
            acc_word:  어절 정확도
            f_score:  f-score
        Returns:
            현재 step이 best 성능을 나타냈는 지 여부
        """
        assert check_id == self.cfg.step // self.cfg.check_step
        avg_loss_train = sum(loss_trains) / len(loss_trains)
        loss_trains.clear()
        self.loss_trains.append(avg_loss_train)
        self.loss_devs.append(avg_loss_dev)
        self.acc_chars.append(acc_char)
        self.acc_words.append(acc_word)
        self.f_scores.append(f_score)
        self.learning_rates.append(self.cfg.learning_rate)
        is_best = self._is_best()
        is_best_str = 'BEST' if is_best else '< {:.4f}'.format(max(self.f_scores))
        tqdm.write('    [Los trn]  [Los dev]  [Acc chr]  [Acc wrd]  [F-score]           [LR]')
        tqdm.write('    {:9.4f}  {:9.4f}  {:9.4f}  {:9.4f}  {:9.4f} {:8}  {:.8f}'.format( \
                avg_loss_train, avg_loss_dev, acc_char, acc_word, f_score, is_best_str,
                self.cfg.learning_rate))
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(check_id, avg_loss_train, avg_loss_dev,
                                                  acc_char, acc_word, f_score,
                                                  self.cfg.learning_rate), file=self.log_file)
        self.log_file.flush()
        self.sum_wrt.add_scalar('loss-train', avg_loss_train, check_id)
        self.sum_wrt.add_scalar('loss-dev', avg_loss_dev, check_id)
        self.sum_wrt.add_scalar('acc-char', acc_char, check_id)
        self.sum_wrt.add_scalar('acc-word', acc_word, check_id)
        self.sum_wrt.add_scalar('f-score', f_score, check_id)
        self.sum_wrt.add_scalar('learning-rate', self.cfg.learning_rate, check_id)
        return is_best

    def _is_best(self) -> bool:
        """
        이번 step에 가장 좋은 성능을 냈는 지 확인하고 그럴 경우 현재 상태를 저장한다.
        Returns:
            best 여부
        """
        if len(self.f_scores) > 1 and max(self.f_scores[:-1]) >= self.f_scores[-1]:
            return False
        # this step hits new max value
        self.cfg.best_step = self.cfg.step
        self.model.save('{}/model.state'.format(self.cfg.out_dir))
        self._save_optim('{}/optim.state'.format(self.cfg.out_dir))
        with open('{}/config.json'.format(self.cfg.out_dir), 'w', encoding='UTF-8') as fout:
            json.dump(vars(self.cfg), fout, indent=2, sort_keys=True)
        return True

    def _save_optim(self, path: str):
        """
        save optimizer parameters
        Args:
            path:  path
        """
        torch.save(self.optimizer.state_dict(), path)

    def _load_optim(self, path: str, learning_rate: float):
        """
        load optimizer parameters
        Args:
            path:  path
            learning_rate:  learning rate
        """
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.optimizer.load_state_dict(state_dict)
        self.optimizer.param_groups[0]['lr'] = learning_rate

    def evaluate(self) -> Tuple[float, float, float, float]:
        """
        evaluate f-score
        Returns:
            average dev loss
            character accuracy
            word accuracy
            f-score
        """
        self.model.eval()
        loss_devs = []
        for dev_sent in self.dataset_dev:
            # 만약 spc_dropout이 1.0 이상이면 공백을 전혀 쓰지 않는 것이므로 평가 시에도 적용한다.
            dev_labels, dev_contexts = dev_sent.to_tensor(self.cfg, self.rsc,
                                                          self.cfg.spc_dropout >= 1.0)
            if torch.cuda.is_available():
                dev_labels = dev_labels.cuda()
                dev_contexts = dev_contexts.cuda()
            dev_outputs = self.model(dev_contexts)
            loss_dev = self.criterion(dev_outputs, dev_labels)
            loss_devs.append(loss_dev.item())
            _, predicts = F.softmax(dev_outputs, dim=1).max(1)
            pred_tags = [self.rsc.vocab_out[t.item()] for t in predicts]
            pred_sent = copy.deepcopy(dev_sent)
            pred_sent.set_pos_result(pred_tags, self.rsc.restore_dic)
            self.evaler.count(dev_sent, pred_sent)
        avg_loss_dev = sum(loss_devs) / len(loss_devs)
        return (avg_loss_dev, ) + self.evaler.evaluate()
