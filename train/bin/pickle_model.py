#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
pickle trained model (state dict)
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
from array import array
import json
import logging
import os
import pickle
import re

import torch

from khaiii.resource.resource import Resource


#############
# functions #
#############
def _validate_state_dict(cfg: Namespace, rsc: Resource, state_dict: dict):
    """
    validate state dict with config and resource
    Args:
        cfg:  config
        rsc:  resource
        state_dict:  state dict
    """
    def _assert(expected, actual, msg: str):
        """
        assert expected equals actual. if not raise exception
        Args:
            expected:  expected value
            actual:  actual value
            msg:  exception message if both are not equal
        """
        if expected != actual:
            raise ValueError((msg + ': expected: {}, but actual: {}').format(expected, actual))

    for name, tensor in state_dict.items():
        logging.info('%s: %s', name, tensor.size())
        if name == 'embedder.embedding.weight':
            _assert(len(rsc.vocab_in), tensor.size(0), 'invalid input vocab')
            _assert(cfg.embed_dim, tensor.size(1), 'invalid embedding dim')
        if re.match(r'convs\..\.weight', name):
            _assert(cfg.embed_dim, tensor.size(1), 'invalid conv input channel dim')
        if name == 'hidden2tag.bias':
            _assert(len(rsc.vocab_out), tensor.size(0), 'invalid output bias dim')


def _get_embedding(rsc: Resource, state_dict: dict) -> dict:
    """
    get character embedding
    Args:
        rsc:  resource
        state_dict:  state dictionary of model
    Returns:
        embedding data
    """
    data = {}
    chars = [0, ] * 5    # 5 special characters
    for idx in range(5, len(rsc.vocab_in)):
        chars.append(ord(rsc.vocab_in[idx]))
    data['chars'] = array('i', chars)    # [input vocab(char)] * 4(wchar_t)

    embedding = state_dict['embedder.embedding.weight']
    data['weights'] = []
    for row in embedding:
        data['weights'].append(array('f', row))    # [input vocab(char)] * embed_dim * 4(float)
    return data


def _get_linear(name_pfx: str, state_dict: dict) -> dict:
    """
    write weight and bias with given layer name prefix
    Args:
        name_pfx:  layer name prefix
        state_dict:  state dictionary of model
    Returns:
        linear layer data
    """
    data = {}
    data['weight'] = []
    weight = state_dict['{}.weight'.format(name_pfx)]
    for row in weight:
        data['weight'].append(array('f', row))    # output * input * 4
    bias_name = '{}.bias'.format(name_pfx)
    if bias_name in state_dict:
        bias = state_dict[bias_name]
        data['bias'] = array('f', bias)    # output * 4
    return data


def _get_conv(name_pfx: str, kernel_size: int, state_dict: dict) -> dict:
    """
    get convolution module weight and bias with given layer name prefix and suffix(index)
    Args:
        name_pfx:  layer name prefix
        kernel_size:  kernel size
        state_dict:  state dictionary of model
    Returns:
        conv data with given kernel size
    """
    data = {}
    weight = state_dict['{}.{}.weight'.format(name_pfx, kernel_size - 2)]
    weight_t = weight.transpose(1, 2)    # [output chan] * [input chan] * kernel
    #                                      => [output chan] * kernel * [input chan]
    data['channel'] = []
    for ch_out in weight_t:
        data['channel'].append(array('f', ch_out.contiguous().view(-1)))
    bias_name = '{}.{}.bias'.format(name_pfx, kernel_size - 2)
    if bias_name in state_dict:
        bias = state_dict[bias_name]
        data['bias'] = array('f', bias)    # [output chan] * 4
    return data


def _get_data(rsc: Resource, state_dict: dict) -> dict:
    """
    get all data to pickle
    Args:
        rsc:  resource
        state_dict:  state dictionary of model
    Returns:
        all data
    """
    data = {}
    # key: [input vocab(char)] * 4(float)
    # val: [input vocab(char)] * embed_dim * 4(float)
    data['embedding'] = _get_embedding(rsc, state_dict)

    data['convs'] = {}
    for kernel in range(2, 6):
        # weight: [output chan(embed_dim)] * kernel * [input chan(embed_dim)] * 4
        # bias: [output chan] * 4
        data['convs'][kernel] = _get_conv('convs', kernel, state_dict)

    # weight: hidden_dim * [cnn layers * output chan(embed_dim)] * 4
    # bias: hidden_dim * 4
    data['conv2hidden'] = _get_linear('conv2hidden', state_dict)

    # weight: [output vocab(tag)] * hidden_dim * 4
    # bias: [output vocab(tag)] * 4
    data['hidden2tag'] = _get_linear('hidden2tag', state_dict)
    return data


def _load_config(path: str) -> Namespace:
    """
    load config file
    Args:
        path:  path
    Returns:
        config
    """
    cfg_dic = json.load(open(path, 'r', encoding='UTF-8'))
    logging.info('config: %s', json.dumps(cfg_dic, indent=4, sort_keys=True))
    cfg = Namespace()
    for key, val in cfg_dic.items():
        if key not in {'cutoff', 'embed_dim', 'hidden_dim', 'model_id', 'model_name', 'rsc_src',
                       'window'}:
            continue
        setattr(cfg, key, val)
    return cfg


def _load_resource(cfg: Namespace, rsc_src: str) -> Resource:
    """
    load resources
    Args:
        cfg:  config
        rsc_src:  resource source dir
    Returns:
        Resource object
    """
    cwd = os.path.realpath(os.getcwd())
    train_dir = os.path.realpath('{}/../../train'.format(rsc_src))
    if cwd != train_dir:
        os.chdir(train_dir)
    rsc = Resource(cfg)
    if cwd != train_dir:
        os.chdir(cwd)
    return rsc


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    cfg = _load_config('{}/config.json'.format(args.in_dir))
    rsc = _load_resource(cfg, args.rsc_src)
    state_dict = torch.load('{}/model.state'.format(args.in_dir),
                            map_location=lambda storage, loc: storage)
    _validate_state_dict(cfg, rsc, state_dict)
    data = _get_data(rsc, state_dict)

    config_path = '{}/{}.config.json'.format(args.rsc_src, args.model_size)
    with open(config_path, 'w', encoding='UTF-8') as fout:
        json.dump(vars(cfg), fout, indent=4, sort_keys=True)

    pickle_path = '{}/{}.model.pickle'.format(args.rsc_src, args.model_size)
    with open(pickle_path, 'wb') as fout:
        pickle.dump(data, fout)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='pickle trained model (state dict)')
    parser.add_argument('-i', '--in-dir', help='model dir', metavar='DIR', required=True)
    parser.add_argument('--model-size', help='model size <default: base>',
                        metavar='SIZE', default='base')
    parser.add_argument('--rsc-src', help='resource source dir <default: ../rsc/src>',
                        metavar='DIR', default='../rsc/src')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
